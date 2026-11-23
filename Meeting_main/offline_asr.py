# -*- coding: utf-8 -*-
"""会后离线转写：连续音频切片 + SEACO 字级时间戳对齐 + Pyannote 全覆盖语音检测。

第一性原理升级（2026-11-23）：
1. 消除流式 VAD 漏检（旧系统在 39 分钟远场会议漏掉 25.85% 语音）：
   新增 `detect_speech_spans` / `transcribe_audio_offline` / `recover_uncovered_spans`，
   复用本机 pyannote/segmentation-3.0 的 17ms 语音概率（覆盖率 >99.7%），在语音低谷处切分 ≤15s 连续段。
2. 消除 `_decode_group` 剪碎拼接与 80ms 双垫音回声（导致“提提了”“我我们”）：
   同组多句直接从原始连续音频切出单一连续片段解码，不插人工零静音、不重复垫音。
3. 利用 SEACO 原生字级毫秒时间戳对齐每条 utterance：
   不再按粗暴字数比例腰斩词组（如“如何把咱这 | 小区”），仅在测试假回调无时间戳时回退比例切分。
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from turns import _is_pure_spoken_digits, clean_cjk_spaces

MAX_SEGMENT_MS = 15000
MIN_SPLIT_MS = 7000
MIN_GROUP_MS = 4000
ADJACENT_GAP_MS = 600
SPEAKER_JOIN_GAP_MS = 2000
SHORT_GROUP_BRIDGE_GAP_MS = 1500
PAD_MS = 80
SPEECH_PAD_MS = 150
SPEECH_GAP_MS = 400
PACK_GAP_MS = 1200
SPEECH_PROB_TH = 0.45
UTT_PAUSE_SPLIT_MS = 650
UTT_MAX_DUR_MS = 10000
MIN_CLIP_MS = 200
LAST_CHUNK_MERGE_MS = 500
INTER_UTT_SILENCE_MAX_MS = 200

AGC_MIN_RMS = 0.002
AGC_MAX_RMS = 0.04
AGC_TARGET_RMS = 0.05
AGC_PEAK_LIMIT = 0.95

_TRAILING_PUNCT = "，。！？、；："

_model = None
_model_lock = threading.Lock()
_osd_sess = None
_osd_lock = threading.Lock()


def _load_asr():
    """延迟加载离线 ASR (Fun-ASR-Nano 优先，缺失时由 get_models 回退 SEACO)。"""
    global _model
    with _model_lock:
        if _model is None:
            from web_server import get_models
            _model = get_models(need_offline=True)["asr"]
        return _model


def _load_osd_session():
    """延迟加载并缓存 pyannote/segmentation-3.0 ONNX 会话；缺失时返回 None。"""
    global _osd_sess
    with _osd_lock:
        if _osd_sess is not None:
            return _osd_sess
        try:
            import onnxruntime as ort
            from offline_diarizer import OSD_ONNX
            if not Path(OSD_ONNX).is_file():
                return None
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail] or avail
            _osd_sess = ort.InferenceSession(str(OSD_ONNX), providers=providers)
            return _osd_sess
        except Exception:  # noqa: BLE001
            return None


def _parse_asr_result(raw_result: Any) -> Tuple[str, Optional[List[Tuple[int, int]]]]:
    """兼容纯字符串（单测假回调）与 (text, timestamps) 二元组（真实 SEACO 输出）。"""
    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        text, ts = raw_result
        return str(text or ""), ts if isinstance(ts, list) else None
    if isinstance(raw_result, dict):
        text = str(raw_result.get("text") or "")
        ts = raw_result.get("timestamp")
        return text, ts if isinstance(ts, list) else None
    return str(raw_result or ""), None


def _join_tokens(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    return clean_cjk_spaces(" ".join(tokens))


def _assign_tokens_by_timestamp(
    raw_text: str,
    timestamps: Sequence[Sequence[int]],
    clip_start_ms: int,
    members: Sequence[Dict],
) -> Optional[List[str]]:
    """把 SEACO 的逐字毫秒时间戳映射到对应 utterance，避免按比例腰斩词组。"""
    tokens = raw_text.strip().split()
    if not tokens or len(tokens) != len(timestamps) or not members:
        return None
    buckets: List[List[str]] = [[] for _ in members]
    for tok, span in zip(tokens, timestamps):
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            return None
        mid_ms = clip_start_ms + 0.5 * (float(span[0]) + float(span[1]))
        chosen = 0
        best_dist = float("inf")
        for idx, utt in enumerate(members):
            s_ms, e_ms = float(utt["start_ms"]), float(utt["end_ms"])
            if s_ms <= mid_ms <= e_ms:
                chosen = idx
                best_dist = 0.0
                break
            dist = (s_ms - mid_ms) if mid_ms < s_ms else (mid_ms - e_ms)
            if dist < best_dist:
                best_dist = dist
                chosen = idx
        buckets[chosen].append(tok)
    return [_join_tokens(b) for b in buckets]


def _tokens_to_utterances(
    raw_text: str,
    timestamps: Optional[Sequence[Sequence[int]]],
    clip_start_ms: int,
    clip_end_ms: int,
    *,
    pause_split_ms: int = UTT_PAUSE_SPLIT_MS,
    max_utt_ms: int = UTT_MAX_DUR_MS,
) -> List[Dict]:
    """把一段连续 SEACO 解码输出按字间真实停顿切分为自然句子（带精确毫秒起止）。"""
    text_clean = clean_cjk_spaces(raw_text)
    if not text_clean:
        return []
    tokens = raw_text.strip().split()
    if not timestamps or len(tokens) != len(timestamps):
        return [{
            "start_ms": int(clip_start_ms),
            "end_ms": int(max(clip_start_ms + 200, clip_end_ms)),
            "text": text_clean,
        }]

    out: List[Dict] = []
    cur_tokens: List[str] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    for tok, span in zip(tokens, timestamps):
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            continue
        t_s = clip_start_ms + int(span[0])
        t_e = clip_start_ms + int(span[1])
        if cur_start is None:
            cur_tokens = [tok]
            cur_start = t_s
            cur_end = max(t_s, t_e)
            continue
        gap = t_s - (cur_end or t_s)
        dur = t_e - cur_start
        # 遇明显停顿或已达 10s 且有轻微换气（>= 180ms）时自然切句
        if gap >= pause_split_ms or (dur >= max_utt_ms and gap >= 180) or dur >= max_utt_ms + 4000:
            joined = _join_tokens(cur_tokens)
            if joined:
                out.append({
                    "start_ms": int(cur_start),
                    "end_ms": int(max(cur_start + 150, cur_end or cur_start)),
                    "text": joined,
                })
            cur_tokens = [tok]
            cur_start = t_s
            cur_end = max(t_s, t_e)
        else:
            cur_tokens.append(tok)
            cur_end = max(cur_end or t_e, t_e)

    if cur_tokens and cur_start is not None:
        joined = _join_tokens(cur_tokens)
        if joined:
            out.append({
                "start_ms": int(cur_start),
                "end_ms": int(max(cur_start + 150, cur_end or cur_start)),
                "text": joined,
            })
    return out


def detect_speech_spans(
    audio: np.ndarray,
    sr: int,
    *,
    max_segment_ms: int = MAX_SEGMENT_MS,
    speech_thresh: float = SPEECH_PROB_TH,
    pad_ms: int = SPEECH_PAD_MS,
    gap_ms: int = SPEECH_GAP_MS,
    pack_gap_ms: int = PACK_GAP_MS,
    vad_model: Any = None,
) -> List[Tuple[int, int]]:
    """全覆盖语音区检测：优先 pyannote 17ms 语音概率（在静音低谷切分），无 ONNX 时回退离线 VAD。"""
    dur_ms = int(len(audio) * 1000 / sr)
    if dur_ms < MIN_CLIP_MS:
        return []

    sess = _load_osd_session()
    if sess is not None:
        from offline_diarizer import DEFAULT_NORMALIZE_PEAK, _normalize_audio
        from run_3dspeaker_pipeline import osd_probs

        norm_audio, tmp_norm = _normalize_audio(audio, sr, DEFAULT_NORMALIZE_PEAK)
        if tmp_norm:
            try:
                os.unlink(tmp_norm)
            except OSError:
                pass
        probs, valid_mask, frame_step_s = osd_probs(sess, norm_audio)
        probs = probs[valid_mask]
        if len(probs) == 0:
            return []
        speech_prob = 1.0 - probs[:, 0]
        active = speech_prob >= speech_thresh

        raw_segs: List[Tuple[int, int]] = []
        in_speech = False
        seg_start = 0
        for idx, flag in enumerate(active):
            if flag and not in_speech:
                seg_start = idx
                in_speech = True
            elif not flag and in_speech:
                raw_segs.append((
                    int(seg_start * frame_step_s * 1000),
                    int(idx * frame_step_s * 1000),
                ))
                in_speech = False
        if in_speech:
            raw_segs.append((
                int(seg_start * frame_step_s * 1000),
                min(dur_ms, int(len(active) * frame_step_s * 1000)),
            ))

        # 合并短间隙 + 加边界 pad
        padded = sorted((max(0, s - pad_ms), min(dur_ms, e + pad_ms)) for s, e in raw_segs if e > s)
        merged: List[Tuple[int, int]] = []
        for s, e in padded:
            if merged and s <= merged[-1][1] + gap_ms:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # 超长段在语音概率最低的低谷处切分，避免腰斩正在说的词
        chunks: List[Tuple[int, int]] = []
        for s, e in merged:
            cur = s
            while e - cur > max_segment_ms:
                lo_f = max(0, int((cur + MIN_SPLIT_MS) / 1000.0 / frame_step_s))
                hi_f = min(len(speech_prob) - 1, int((cur + max_segment_ms) / 1000.0 / frame_step_s))
                if hi_f > lo_f:
                    best_f = lo_f + int(np.argmin(speech_prob[lo_f:hi_f + 1]))
                    cut = int(best_f * frame_step_s * 1000)
                else:
                    cut = cur + max_segment_ms
                if cut <= cur + 1000:
                    cut = min(e, cur + max_segment_ms)
                chunks.append((cur, cut))
                cur = cut
            if e - cur >= MIN_CLIP_MS:
                chunks.append((cur, e))

        # 相邻短段在不超 15s 且间隔 <= pack_gap_ms 时合并为连续窗
        packed: List[Tuple[int, int]] = []
        for s, e in chunks:
            if packed and (s - packed[-1][1] <= pack_gap_ms) and (e - packed[-1][0] <= max_segment_ms):
                packed[-1] = (packed[-1][0], e)
            else:
                packed.append((s, e))
        return packed

    # 回退离线 FSMN-VAD
    if vad_model is None:
        try:
            from web_server import get_models
            vad_model = get_models(need_stream=False, need_offline=False).get("vad")
        except Exception:  # noqa: BLE001
            vad_model = None
    if vad_model is not None:
        import soundfile as sf
        fd, tmp_wav = tempfile.mkstemp(prefix="offline_vad_", suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp_wav, audio, sr, subtype="PCM_16")
            vres = vad_model.generate(input=tmp_wav)
            segs = vres[0].get("value", []) if vres else []
        finally:
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass
        padded = sorted((max(0, int(s) - pad_ms), min(dur_ms, int(e) + pad_ms)) for s, e in segs if e > s)
        packed = []
        for s, e in padded:
            for lo, hi in _split_ranges(s, e, max_segment_ms):
                if hi - lo < MIN_CLIP_MS:
                    continue
                if packed and (lo - packed[-1][1] <= pack_gap_ms) and (hi - packed[-1][0] <= max_segment_ms):
                    packed[-1] = (packed[-1][0], hi)
                else:
                    packed.append((lo, hi))
        return packed

    return _split_ranges(0, dur_ms, max_segment_ms)


def _split_ranges(start_ms: int, end_ms: int, max_ms: int) -> List[tuple]:
    total = max(0, end_ms - start_ms)
    if total <= max_ms:
        return [(start_ms, end_ms)]
    count = (total + max_ms - 1) // max_ms
    step = total / count
    return [(int(start_ms + i * step), int(start_ms + (i + 1) * step))
            for i in range(count)]


def _utt_duration(utt: Dict) -> int:
    return max(0, int(utt["end_ms"]) - int(utt["start_ms"]))


def _utt_weight(utt: Dict) -> int:
    """优先按原句 raw_text 有效字符数分配切分权重，无文本时退回音频时长。"""
    ref = clean_cjk_spaces(utt.get("raw_text") or "")
    core_len = len(ref.rstrip(_TRAILING_PUNCT + ",.!? "))
    if core_len > 0:
        return core_len * 200
    return max(1, _utt_duration(utt))


def _restore_trailing_punct(piece: str, utt: Dict) -> str:
    """[E4] 仅当原句 raw_text (非 LLM 润色稿) 自带标点时才继承句尾标点。"""
    piece = clean_cjk_spaces(piece)
    if not piece:
        return ""
    if piece[-1] in _TRAILING_PUNCT or piece[-1] in ",.!?":
        return piece
    raw_ref = clean_cjk_spaces(utt.get("raw_text") or "")
    if raw_ref and raw_ref[-1] in _TRAILING_PUNCT:
        return piece + raw_ref[-1]
    return piece


def _concat_decoded_chunks(parts: Sequence[str]) -> str:
    """[E2] 拼接长段切分后的解码片段：直接无缝连接并清理 CJK 空格，不在小数/英文词中硬插空格。"""
    valid = [clean_cjk_spaces(p) for p in parts if p and p.strip()]
    if not valid:
        return ""
    return clean_cjk_spaces("".join(valid))


def _agc(clip: np.ndarray, stats: Optional[Dict] = None) -> np.ndarray:
    """RMS 自动增益：只处理 0.002 < rms < 0.04 的弱音片段。"""
    if clip.size == 0:
        return clip
    mono = clip.astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(np.square(mono))))
    if not (AGC_MIN_RMS < rms < AGC_MAX_RMS):
        return clip

    gain = AGC_TARGET_RMS / rms
    peak = float(np.max(np.abs(mono)))
    limited = False
    if peak > 0:
        peak_gain = AGC_PEAK_LIMIT / peak
        if gain > peak_gain:
            gain = peak_gain
            limited = True
    if stats is not None:
        stats["agc_clips"] = stats.get("agc_clips", 0) + 1
        if limited:
            stats["agc_peak_limited"] = stats.get("agc_peak_limited", 0) + 1
    return (mono * gain).astype(np.float32)


def _clip(audio: np.ndarray, sr: int, start_ms: int, end_ms: int) -> Optional[np.ndarray]:
    lo = max(0, int((start_ms - PAD_MS) * sr / 1000))
    hi = min(len(audio), int((end_ms + PAD_MS) * sr / 1000))
    if hi - lo < int(MIN_CLIP_MS * sr / 1000):
        return None
    return audio[lo:hi].copy()


def _group_utterances(utterances: Sequence[Dict]) -> List[List[Dict]]:
    """把短句贪心合并成 4s~15s 的组。

    [A4] 若相邻两句是纯逐位报号串 (如 '零零六' 与 '零幺三')，不把它们强行拼成一个无间隔组，
    防止 SEACO 解码后比例切分把不同组号码切错位。
    """
    groups: List[List[Dict]] = []
    current: List[Dict] = []
    for utt in utterances:
        if not current:
            current = [utt]
            continue
        prev = current[-1]
        # 报号短句各自独立解码，保留真实组边界
        if (_is_pure_spoken_digits(prev.get("raw_text") or "")
                and _is_pure_spoken_digits(utt.get("raw_text") or "")):
            groups.append(current)
            current = [utt]
            continue

        gap = int(utt["start_ms"]) - int(prev["end_ms"])
        same_spk = (
            utt.get("spk_label") is not None
            and utt.get("spk_label") == prev.get("spk_label")
        )
        no_labels = (utt.get("spk_label") is None
                     and prev.get("spk_label") is None)
        join = ((same_spk and gap <= SPEAKER_JOIN_GAP_MS)
                or (no_labels and gap < ADJACENT_GAP_MS))
        current_duration = sum(_utt_duration(item) for item in current)
        if (not join and current_duration < MIN_GROUP_MS
                and gap < SHORT_GROUP_BRIDGE_GAP_MS
                and (no_labels or same_spk)):
            join = True
        if join and current_duration + _utt_duration(utt) <= MAX_SEGMENT_MS:
            current.append(utt)
        else:
            groups.append(current)
            current = [utt]
    if current:
        groups.append(current)
    return groups


def _split_text(text: str, weights: Sequence[int]) -> List[str]:
    """按权重把清理空格后的解码文本切回各 utt。"""
    text = clean_cjk_spaces(text)
    count = len(weights)
    if count <= 1:
        return [text]
    total = sum(weights) or count
    bounds = [0]
    acc = 0
    for weight in weights[:-1]:
        acc += weight
        bounds.append(int(round(len(text) * acc / total)))
    bounds.append(len(text))
    for index in range(1, len(bounds)):
        bounds[index] = max(bounds[index], bounds[index - 1])
    return [text[bounds[i]:bounds[i + 1]] for i in range(count)]


def _decode_single(
    audio: np.ndarray,
    sr: int,
    utt: Dict,
    caller: Callable[[str], Any],
    tmp_path: str,
    max_segment_ms: int,
    use_agc: bool,
    stats: Dict,
) -> str:
    import soundfile as sf

    parts: List[str] = []
    start_ms, end_ms = int(utt["start_ms"]), int(utt["end_ms"])
    for lo_ms, hi_ms in _split_ranges(start_ms, end_ms, max_segment_ms):
        clip = _clip(audio, sr, lo_ms, hi_ms)
        if clip is None:
            continue
        if use_agc:
            clip = _agc(clip, stats)
        sf.write(tmp_path, clip, sr, subtype="PCM_16")
        raw_txt, _ = _parse_asr_result(caller(tmp_path))
        text = clean_cjk_spaces(raw_txt)
        stats["segments"] = stats.get("segments", 0) + 1
        if text:
            parts.append(text)
    joined = _concat_decoded_chunks(parts)
    return _restore_trailing_punct(joined, utt)


def _decode_group(
    audio: np.ndarray,
    sr: int,
    group: Sequence[Dict],
    caller: Callable[[str], Any],
    tmp_path: str,
    max_segment_ms: int,
    use_agc: bool,
    stats: Dict,
) -> Dict[str, Dict]:
    """连续切片解码一组同说话人句子，优先按 SEACO 字级毫秒时间戳精确归位。"""
    import soundfile as sf

    if len(group) == 1:
        utt = group[0]
        text = _decode_single(audio, sr, utt, caller, tmp_path,
                              max_segment_ms, use_agc, stats)
        if not text:
            return {}
        return {utt["utt_id"]: {
            "text": text,
            "start_ms": int(utt["start_ms"]),
            "end_ms": int(utt["end_ms"]),
            "grouped_cont": False,
        }}

    members: List[Dict] = []
    for utt in group:
        s_ms, e_ms = int(utt["start_ms"]), int(utt["end_ms"])
        if e_ms - s_ms + 2 * PAD_MS >= MIN_CLIP_MS:
            members.append(utt)
    if not members:
        return {}

    # 第一性原理修复：直接从连续原始音频切一整段 [span_start - PAD, span_end + PAD]，
    # 彻底消除逐句 80ms 双垫音回声（“提提了”“我我们”）与人工零静音断崖，同时保留句间弱音字头。
    span_start_ms = min(int(u["start_ms"]) for u in members)
    span_end_ms = max(int(u["end_ms"]) for u in members)
    clip_start_ms = max(0, span_start_ms - PAD_MS)
    audio_dur_ms = int(len(audio) * 1000 / sr)
    clip_end_ms = min(audio_dur_ms, span_end_ms + PAD_MS)

    if clip_end_ms - clip_start_ms < MIN_CLIP_MS:
        return {}

    # 单段 <= max_segment_ms（绝大多数情况）：一次连续解码 + 字级时间戳精确归位
    if clip_end_ms - clip_start_ms <= max_segment_ms + LAST_CHUNK_MERGE_MS:
        lo = int(clip_start_ms * sr / 1000)
        hi = int(clip_end_ms * sr / 1000)
        clip = audio[lo:hi].copy()
        if use_agc:
            clip = _agc(clip, stats)
        sf.write(tmp_path, clip, sr, subtype="PCM_16")
        raw_txt, timestamps = _parse_asr_result(caller(tmp_path))
        stats["segments"] = stats.get("segments", 0) + 1
        text_clean = clean_cjk_spaces(raw_txt)
        if not text_clean:
            return {}

        pieces = None
        ts_aligned = False
        if timestamps:
            pieces = _assign_tokens_by_timestamp(raw_txt, timestamps, clip_start_ms, members)
            ts_aligned = pieces is not None
        if pieces is None:
            weights = [_utt_weight(utt) for utt in members]
            pieces = _split_text(text_clean, weights)

        out: Dict[str, Dict] = {}
        for idx, (utt, piece) in enumerate(zip(members, pieces)):
            if idx == len(members) - 1:
                piece = _restore_trailing_punct(piece, utt)
            out[utt["utt_id"]] = {
                "text": piece,
                "start_ms": int(utt["start_ms"]),
                "end_ms": int(utt["end_ms"]),
                # 时间戳对齐时每个 utt 拿到的是自己时间段内的完整词，不再属于被比例拦腰切开的残片
                "grouped_cont": (idx > 0) and (not ts_aligned),
            }
        return out

    # 超长 span 回退：按连续范围切块，仍优先汇总时间戳精确归位
    all_tokens: List[str] = []
    all_ts: List[List[int]] = []
    ts_ok = True
    parts: List[str] = []
    for lo_ms, hi_ms in _split_ranges(clip_start_ms, clip_end_ms, max_segment_ms):
        lo = int(lo_ms * sr / 1000)
        hi = int(hi_ms * sr / 1000)
        chunk = audio[lo:hi].copy()
        if len(chunk) < int(MIN_CLIP_MS * sr / 1000):
            continue
        if use_agc:
            chunk = _agc(chunk, stats)
        sf.write(tmp_path, chunk, sr, subtype="PCM_16")
        raw_txt, timestamps = _parse_asr_result(caller(tmp_path))
        stats["segments"] = stats.get("segments", 0) + 1
        clean_part = clean_cjk_spaces(raw_txt)
        if clean_part:
            parts.append(clean_part)
        toks = raw_txt.strip().split()
        if ts_ok and timestamps and len(toks) == len(timestamps):
            for tok, sp in zip(toks, timestamps):
                all_tokens.append(tok)
                all_ts.append([int(sp[0]) + (lo_ms - clip_start_ms), int(sp[1]) + (lo_ms - clip_start_ms)])
        elif toks:
            ts_ok = False

    text = _concat_decoded_chunks(parts)
    if not text:
        return {}

    pieces = None
    ts_aligned = False
    if ts_ok and all_tokens:
        pieces = _assign_tokens_by_timestamp(" ".join(all_tokens), all_ts, clip_start_ms, members)
        ts_aligned = pieces is not None
    if pieces is None:
        weights = [_utt_weight(utt) for utt in members]
        pieces = _split_text(text, weights)

    out = {}
    for idx, (utt, piece) in enumerate(zip(members, pieces)):
        if idx == len(members) - 1:
            piece = _restore_trailing_punct(piece, utt)
        out[utt["utt_id"]] = {
            "text": piece,
            "start_ms": int(utt["start_ms"]),
            "end_ms": int(utt["end_ms"]),
            "grouped_cont": (idx > 0) and (not ts_aligned),
        }
    return out


def _make_seaco_caller(hotwords: Optional[Union[str, List[str]]], stats: Dict) -> Callable[[str], Tuple[str, Optional[List]]]:
    model = _load_asr()
    hotword_state = {"enabled": hotwords is not None}

    def caller(path: str) -> Tuple[str, Optional[List]]:
        if hotword_state["enabled"]:
            try:
                result = model.generate(input=path, hotword=hotwords)
            except Exception:  # noqa: BLE001
                hotword_state["enabled"] = False
                stats["hotword_fallbacks"] = stats.get("hotword_fallbacks", 0) + 1
                result = model.generate(input=path)
        else:
            result = model.generate(input=path)
        if not result:
            return "", None
        r0 = result[0]
        return (r0.get("text") or ""), r0.get("timestamp")

    return caller


def transcribe_utterances(
    audio_path,
    utterances: List[Dict],
    *,
    asr_call: Optional[Callable[[str], Any]] = None,
    max_segment_ms: int = MAX_SEGMENT_MS,
    agc: bool = True,
    group_short: bool = True,
    hotwords: Optional[Union[str, List[str]]] = None,
    on_progress: Optional[Callable[[Dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
    on_stats: Optional[Callable[[Dict], None]] = None,
) -> Dict[str, Dict]:
    """返回 {utt_id: {"text", "start_ms", "end_ms", "grouped_cont"}}；失败/空段不出现。"""
    import soundfile as sf

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = data[:, 0]

    stats: Dict = {
        "utterances": len(utterances),
        "groups": 0,
        "segments": 0,
        "agc_clips": 0,
        "agc_peak_limited": 0,
        "failed_groups": 0,
        "empty_groups": 0,
        "hotword_fallbacks": 0,
    }

    caller = asr_call if asr_call is not None else _make_seaco_caller(hotwords, stats)

    out: Dict[str, Dict] = {}
    total = len(utterances)
    done = 0
    tmp = os.path.join(tempfile.gettempdir(), f"offline_asr_{os.getpid()}_{threading.get_ident()}.wav")
    groups = _group_utterances(utterances) if group_short else [[u] for u in utterances]

    try:
        for group in groups:
            if should_abort is not None and should_abort():
                break
            stats["groups"] += 1
            try:
                part = _decode_group(audio, sr, group, caller, tmp,
                                     max_segment_ms, agc, stats)
            except Exception:  # noqa: BLE001
                stats["failed_groups"] += 1
                part = {}
            if part:
                out.update(part)
            else:
                stats["empty_groups"] += 1

            done += len(group)
            if on_progress is not None:
                on_progress({
                    "t": "progress", "stage": "final_asr",
                    "done": done, "total": total,
                    "msg": "最终稿离线转写中...",
                })
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass

    if on_stats is not None:
        on_stats(stats)
    return out


def transcribe_audio_offline(
    audio_path,
    *,
    asr_call: Optional[Callable[[str], Any]] = None,
    spans: Optional[Sequence[Tuple[int, int]]] = None,
    vad_model: Any = None,
    max_segment_ms: int = MAX_SEGMENT_MS,
    agc: bool = True,
    hotwords: Optional[Union[str, List[str]]] = None,
    on_progress: Optional[Callable[[Dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> Tuple[List[Dict], int]:
    """直通离线高精度转写（用于导入音频文件）：绕过流式漏检与废稿，一次直出精确时间戳句表。

    返回 (utterances, duration_ms)，每条含 utt_id / start_ms / end_ms / raw_text / final_text。
    """
    import soundfile as sf

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = data[:, 0]
    duration_ms = int(len(audio) * 1000 / sr)
    if duration_ms < MIN_CLIP_MS:
        return [], duration_ms

    speech_spans = list(spans) if spans is not None else detect_speech_spans(
        audio, sr, max_segment_ms=max_segment_ms, vad_model=vad_model,
    )
    if not speech_spans:
        return [], duration_ms

    stats: Dict = {"segments": 0, "agc_clips": 0, "agc_peak_limited": 0}
    caller = asr_call if asr_call is not None else _make_seaco_caller(hotwords, stats)

    total = len(speech_spans)
    tmp = os.path.join(tempfile.gettempdir(), f"offline_direct_{os.getpid()}_{threading.get_ident()}.wav")
    utterances: List[Dict] = []
    utt_idx = 0

    try:
        for i, (s_ms, e_ms) in enumerate(speech_spans, 1):
            if should_abort is not None and should_abort():
                break
            lo = max(0, int(s_ms * sr / 1000))
            hi = min(len(audio), int(e_ms * sr / 1000))
            if hi - lo < int(MIN_CLIP_MS * sr / 1000):
                continue
            clip = audio[lo:hi].copy()
            if agc:
                clip = _agc(clip, stats)
            sf.write(tmp, clip, sr, subtype="PCM_16")
            try:
                raw_txt, timestamps = _parse_asr_result(caller(tmp))
            except Exception:  # noqa: BLE001
                raw_txt, timestamps = "", None

            pieces = _tokens_to_utterances(raw_txt, timestamps, int(s_ms), int(e_ms))
            for item in pieces:
                utt_idx += 1
                utterances.append({
                    "utt_id": f"utt_{utt_idx:04d}",
                    "start_ms": item["start_ms"],
                    "end_ms": item["end_ms"],
                    "raw_text": item["text"],
                    "final_text": item["text"],
                })

            if on_progress is not None:
                on_progress({
                    "t": "progress",
                    "stage": "final_asr",
                    "done": i,
                    "total": total,
                    "msg": f"正在高精度转写 ({i}/{total})...",
                })
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass

    return utterances, duration_ms


def recover_uncovered_spans(
    audio_path,
    existing_utterances: Sequence[Dict],
    *,
    asr_call: Optional[Callable[[str], Any]] = None,
    spans: Optional[Sequence[Tuple[int, int]]] = None,
    vad_model: Any = None,
    min_uncovered_ms: int = 600,
    max_segment_ms: int = MAX_SEGMENT_MS,
    agc: bool = True,
    hotwords: Optional[Union[str, List[str]]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> List[Dict]:
    """会后漏字打捞：找出流式 VAD 漏检的语音空洞并补转写，返回按时间排序的新增句子。"""
    import soundfile as sf

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = data[:, 0]
    dur_ms = int(len(audio) * 1000 / sr)
    if dur_ms < min_uncovered_ms:
        return []

    speech_spans = list(spans) if spans is not None else detect_speech_spans(
        audio, sr, max_segment_ms=max_segment_ms, vad_model=vad_model,
    )
    if not speech_spans:
        return []

    # 计算已有流式句子的合并覆盖区间
    cov = sorted(
        (max(0, int(u["start_ms"]) - PAD_MS), min(dur_ms, int(u["end_ms"]) + PAD_MS))
        for u in existing_utterances
        if int(u.get("end_ms") or 0) > int(u.get("start_ms") or 0)
    )
    merged_cov: List[Tuple[int, int]] = []
    for a, b in cov:
        if merged_cov and a <= merged_cov[-1][1]:
            merged_cov[-1] = (merged_cov[-1][0], max(merged_cov[-1][1], b))
        else:
            merged_cov.append((a, b))

    # 减去已覆盖区间，只保留长度 >= min_uncovered_ms 的真实漏检语音段
    missing: List[Tuple[int, int]] = []
    for s, e in speech_spans:
        cur = s
        for a, b in merged_cov:
            if b <= cur:
                continue
            if a >= e:
                break
            if a - cur >= min_uncovered_ms:
                missing.append((cur, a))
            cur = max(cur, b)
        if e - cur >= min_uncovered_ms:
            missing.append((cur, e))

    if not missing:
        return []

    recovered, _ = transcribe_audio_offline(
        audio_path,
        asr_call=asr_call,
        spans=missing,
        max_segment_ms=max_segment_ms,
        agc=agc,
        hotwords=hotwords,
        should_abort=should_abort,
    )
    for idx, item in enumerate(recovered, 1):
        item["utt_id"] = f"utt_rec_{idx:04d}"
    return recovered
