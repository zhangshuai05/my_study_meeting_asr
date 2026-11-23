# -*- coding: utf-8 -*-
"""会后离线二遍转写：按 VAD 句子重转写，产出最终稿。

2026-10-19 审计整改升级：
1. [A4] 调用升级版 `clean_cjk_spaces`，清除 SEACO 分词空格的同时保留报号数字串边界。
2. [E2] 同一长段切片解码拼接时使用 `_concat_Decoded_chunks`，不在英文单词或小数 (`3.5`) 中间误插空格。
3. [E3] 组内多句音频拼接时保留最多 `200ms` 的句间自然静音间隔，不把两句话死贴在一起，保留声学停顿线索。
4. [E4] `_restore_trailing_punct` 仅从原始 `raw_text` (而非流式 LLM 猜测的 `polished_text`) 恢复真实标点。
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from turns import _is_pure_spoken_digits, clean_cjk_spaces

MAX_SEGMENT_MS = 15000
MIN_GROUP_MS = 4000
ADJACENT_GAP_MS = 600
SPEAKER_JOIN_GAP_MS = 2000
SHORT_GROUP_BRIDGE_GAP_MS = 1500
PAD_MS = 80
MIN_CLIP_MS = 200
LAST_CHUNK_MERGE_MS = 500
INTER_UTT_SILENCE_MAX_MS = 200   # [E3] 组内拼接时保留的最大自然静音间隔

AGC_MIN_RMS = 0.002
AGC_MAX_RMS = 0.04
AGC_TARGET_RMS = 0.05
AGC_PEAK_LIMIT = 0.95

_TRAILING_PUNCT = "，。！？、；："

_model = None
_model_lock = threading.Lock()


def _load_asr():
    """延迟加载离线 ASR (Fun-ASR-Nano 优先，缺失时由 get_models 回退 SEACO)。"""
    global _model
    with _model_lock:
        if _model is None:
            from web_server import get_models
            _model = get_models(need_offline=True)["asr"]
        return _model


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
    caller: Callable[[str], str],
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
        text = clean_cjk_spaces(caller(tmp_path) or "")
        stats["segments"] = stats.get("segments", 0) + 1
        if text:
            parts.append(text)
    joined = _concat_decoded_chunks(parts)
    return _restore_trailing_punct(joined, utt)


def _decode_group(
    audio: np.ndarray,
    sr: int,
    group: Sequence[Dict],
    caller: Callable[[str], str],
    tmp_path: str,
    max_segment_ms: int,
    use_agc: bool,
    stats: Dict,
) -> Dict[str, Dict]:
    """解码一个组，返回 {utt_id: {text, start_ms, end_ms, grouped_cont}}。"""
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

    members: List[tuple] = []
    concat_arrays: List[np.ndarray] = []
    prev_end_ms: Optional[int] = None
    for utt in group:
        s_ms, e_ms = int(utt["start_ms"]), int(utt["end_ms"])
        clip = _clip(audio, sr, s_ms, e_ms)
        if clip is None:
            continue
        # [E3] 若两句之间存在自然停顿，保留最多 200ms 静音缓冲，避免前句尾音与后句字头死贴
        if prev_end_ms is not None and s_ms > prev_end_ms:
            sil_ms = min(INTER_UTT_SILENCE_MAX_MS, s_ms - prev_end_ms)
            if sil_ms >= 40:
                concat_arrays.append(np.zeros(int(sil_ms * sr / 1000), dtype=np.float32))
        proc_clip = _agc(clip, stats) if use_agc else clip
        members.append((utt, proc_clip))
        concat_arrays.append(proc_clip)
        prev_end_ms = e_ms

    if not members:
        return {}
    joined = np.concatenate(concat_arrays)

    max_samples = max(1, int(max_segment_ms * sr / 1000))
    chunks = [joined[i:i + max_samples] for i in range(0, len(joined), max_samples)]
    merge_samples = int(LAST_CHUNK_MERGE_MS * sr / 1000)
    if len(chunks) >= 2 and len(chunks[-1]) < merge_samples:
        chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
        chunks.pop()

    parts: List[str] = []
    for chunk in chunks:
        sf.write(tmp_path, chunk, sr, subtype="PCM_16")
        text = clean_cjk_spaces(caller(tmp_path) or "")
        stats["segments"] = stats.get("segments", 0) + 1
        if text:
            parts.append(text)
    text = _concat_decoded_chunks(parts)
    if not text:
        return {}
    weights = [_utt_weight(utt) for utt, _ in members]
    pieces = _split_text(text, weights)
    out: Dict[str, Dict] = {}
    for idx, ((utt, _), piece) in enumerate(zip(members, pieces)):
        if idx == len(members) - 1:
            piece = _restore_trailing_punct(piece, utt)
        out[utt["utt_id"]] = {
            "text": piece,
            "start_ms": int(utt["start_ms"]),
            "end_ms": int(utt["end_ms"]),
            "grouped_cont": idx > 0,
        }
    return out


def transcribe_utterances(
    audio_path,
    utterances: List[Dict],
    *,
    asr_call: Optional[Callable[[str], str]] = None,
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

    hotword_state = {"enabled": hotwords is not None and asr_call is None}

    if asr_call is not None:
        caller = asr_call
    else:
        model = _load_asr()

        def caller(path: str) -> str:
            if hotword_state["enabled"]:
                try:
                    result = model.generate(input=path, hotword=hotwords)
                except Exception:  # noqa: BLE001
                    hotword_state["enabled"] = False
                    stats["hotword_fallbacks"] += 1
                    result = model.generate(input=path)
            else:
                result = model.generate(input=path)
            return (result[0].get("text") or "") if result else ""

    out: Dict[str, Dict] = {}
    total = len(utterances)
    done = 0
    tmp = os.path.join(tempfile.gettempdir(), f"offline_asr_{os.getpid()}.wav")
    groups = _group_utterances(utterances) if group_short else [[u] for u in utterances]

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

    if on_stats is not None:
        on_stats(stats)
    return out
