# -*- coding: utf-8 -*-
"""说话人 turn 合并层：utterances 保细粒度，turns 供显示/导出。

设计规则：utterances = 声纹/审计粒度，turns = 显示/导出粒度。
本模块只做纯文本/时间合并，不读写 session 目录，也不依赖 session_store (避免循环导入)。

2026-10-19 审计整改升级：
1. [A4] 保護中文报号/数字串边界 (`clean_cjk_spaces`)：
   - 识别中文数字/报号字 (`零〇一二两三四五六七八九幺洞拐勾`)；当两个中文数字之间在原文本中
     存在空格分隔且属于独立报号组 (例如 `"零 零 六"` 与 `"零 幺 三"` 跨句拼接，或组间多空格)，
     自动保留顿号 `"、"` 或单空格，绝不把 `006 / 013 / 014 / 019` 糊成一串 `零零六零幺三零幺四零幺九`。
2. [D1] 严格保护 `raw_text` 原值：绝不对 `utt["raw_text"]` 执行去空格改写，保证与 `utterances.jsonl` 逐字一致。
3. [E1] 清理 `_END_PUNCT`：移除空格与普通引号，仅将真正的标点符号视为句末标点。
4. [E8] 增加单段 Turn 最大时长 (`MAX_TURN_DURATION_MS = 60000`) 与最大字数 (`MAX_TURN_CHARS = 500`) 软上限，
   防止 10 分钟长独白合成上千字巨型段落撑爆 LLM 4096 token 上限。
5. [A5/C1] 每个 Turn 同时保存 `text` (当前定稿)、`stream_text` (直播润色稿)、`raw_text` (原始流式稿)，
   并准确标记 `text_source` (`"turn_polished"` | `"final_asr"` | `"stream_polished"` | `"raw"`)。
"""
from __future__ import annotations

import re
import threading
from typing import Dict, List, Optional

STREAM_TURN_GAP_MS = 5000      # 流式显示：只有长停顿才换段，避免说话人未知切得太碎
FINAL_TURN_SPLIT_MS = 8000     # 最终合并：同一说话人也在长静音处断开
MAX_TURN_DURATION_MS = 60000   # [E8] 单个 Turn 最大时长软上限 (60秒)，防止长独白段过大导致 LLM 截断
MAX_TURN_CHARS = 500           # [E8] 单个 Turn 最大字符数软上限 (500字)
MIN_SUB_SEGMENT_MS = 100       # sub_turns 拆分后短于此的碎片并入相邻段
PAUSE_COMMA_GAP_MS = 450       # 句间真实停顿达到此阈值才补逗号，避免词中硬插逗号

# [E1] 严禁把空格或普通英文引号放进 _END_PUNCT，避免把"以空格结尾"误判为"已有标点"
_CJK_PUNCT = "，。！？、；：“”‘’（）《》【】…"
_END_PUNCT = "，。！？、；：,.!?…）”’》】"
_TERMINAL_PUNCT = "。！？!?…"

# 中文报号单字集合 (不含十/百/千/万等量词，专门针对 006 / 013 等逐位报号)
_SPOKEN_DIGITS = "零〇一二两三四五六七八九幺洞拐勾"
_RE_SPOKEN_DIGIT_GROUP = re.compile(
    rf"^[{_SPOKEN_DIGITS}]+(?:\s+[{_SPOKEN_DIGITS}]+)*$"
)
# 当 SEACO 输出中两个数字组之间有 >=2 个空格时，代表明显停顿分界
_RE_DIGIT_MULTI_SPACE = re.compile(
    rf"([{_SPOKEN_DIGITS}])\s{{2,}}([{_SPOKEN_DIGITS}])"
)
# 连续 6 位及以上纯逐位报号 (如 "零幺三零幺四零幺九" = 9 位，3 的倍数) 自动按 3 位补顿号
_RE_CONSECUTIVE_TRIPLETS = re.compile(f"^([{_SPOKEN_DIGITS}]{{3}}){{2,}}$")

# 匹配中文字符或中文标点之间的多余空格 (来自 SEACO 分词输出)
_CJK_RANGE = r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]"
_RE_CJK_SPACE_BOTH = re.compile(rf"({_CJK_RANGE})\s+({_CJK_RANGE})")
_RE_CJK_SPACE_LEFT = re.compile(rf"({_CJK_RANGE})\s+([,.:;!?%])")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")


def _is_cjk(ch: str) -> bool:
    """判断单个字符是否属于中日韩统一表意文字或中文标点。"""
    return "\u4e00" <= ch <= "\u9fff" or ch in _CJK_PUNCT


def _is_pure_spoken_digits(text: str) -> bool:
    """判断去掉空白和标点后是否全由逐位报号汉字组成 (如 '零零六'、'零幺三')。"""
    core = re.sub(r"[\s，。！？、；：,.!?]", "", text or "")
    return len(core) >= 2 and all(ch in _SPOKEN_DIGITS for ch in core)


def clean_cjk_spaces(text: Optional[str]) -> str:
    """清除汉字之间的 SEACO 分词空格，同时保护英文单词空格与报号数字组边界 [A4]。"""
    if not text:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    # 1. 若兩個報號數字之間原本有 >=2 個空格，先轉為頓號保護邊界
    s = _RE_DIGIT_MULTI_SPACE.sub(r"\1、\2", s)
    # 2. 清除 CJK 字符之間的單個分詞空格 (執行兩次覆蓋奇偶相鄰字)
    s = _RE_CJK_SPACE_BOTH.sub(r"\1\2", s)
    s = _RE_CJK_SPACE_BOTH.sub(r"\1\2", s)
    s = _RE_CJK_SPACE_LEFT.sub(r"\1\2", s)
    s = _RE_MULTI_SPACE.sub(" ", s).strip()
    # 3. 若整句去掉空格後是 6 位及以上的純 3 位制連續報號 (如 "零幺三零幺四零幺九")，按 3 位一組補頓號
    core_no_punct = s.rstrip(_END_PUNCT)
    tail_punct = s[len(core_no_punct):]
    if _RE_CONSECUTIVE_TRIPLETS.match(core_no_punct):
        groups = [core_no_punct[i:i + 3] for i in range(0, len(core_no_punct), 3)]
        s = "、".join(groups) + tail_punct
    return s


def ensure_terminal_punct(text: str) -> str:
    """确保完整段落末尾带有终结标点 (若末尾是逗号/顿号则替换为句号，若无标点则补句号)。"""
    s = clean_cjk_spaces(text)
    if not s:
        return ""
    if s[-1] in _TERMINAL_PUNCT:
        return s
    if s[-1] in "，、；,;":
        return s[:-1] + ("。" if _is_cjk(s[:-1][-1:] or "中") else ".")
    if s[-1] in _END_PUNCT:
        return s
    return s + ("。" if _is_cjk(s[-1]) else ".")


def join_texts(parts: List[str], *, insert_comma: bool = True) -> str:
    """拼接多个文本片段 (先清理 SEACO 汉字间空格)。

    [A4] 当相邻两段均为纯逐位报号串 (如 '零零六' 与 '零幺三') 且无标点时，
         即使 insert_comma=False 也强制插入顿号 '、'，防止不同报号粘连成一串。
    """
    joined = ""
    prev_was_digits = False
    for part in parts:
        text = clean_cjk_spaces(part)
        if not text:
            continue
        cur_is_digits = _is_pure_spoken_digits(text)
        if joined and joined[-1] not in _END_PUNCT and text[0] not in _END_PUNCT:
            if prev_was_digits and cur_is_digits:
                joined += "、"
            elif _is_cjk(joined[-1]) or _is_cjk(text[0]):
                if insert_comma:
                    joined += "，"
            else:
                joined += " "
        joined += text
        prev_was_digits = cur_is_digits
    return joined


def _split_text_by_weights(text, weights: List[int], *, clean: bool = True):
    """按时间权重把一段文本切给多个子段；None 保持 None。"""
    if text is None:
        return [None] * len(weights)
    s = clean_cjk_spaces(text) if clean else str(text)
    count = len(weights)
    if count <= 1:
        return [s]
    total = sum(weights) or count
    bounds = [0]
    acc = 0
    for weight in weights[:-1]:
        acc += max(1, weight)
        bounds.append(int(round(len(s) * acc / total)))
    bounds.append(len(s))
    for index in range(1, len(bounds)):
        bounds[index] = max(bounds[index], bounds[index - 1])
    return [s[bounds[i]:bounds[i + 1]] for i in range(count)]


def split_utterance_by_sub_turns(
    utt: Dict,
    sub_turns,
    *,
    min_segment_ms: int = MIN_SUB_SEGMENT_MS,
) -> List[Dict]:
    """把一句按上游 `sub_turns` 拆成独立的单人子段。"""
    try:
        start, end = int(utt["start_ms"]), int(utt["end_ms"])
    except (KeyError, TypeError, ValueError):
        return [utt]
    if not sub_turns or end <= start:
        return [utt]

    main_spk, main_label = utt.get("spk"), utt.get("spk_label")
    marks = []
    for item in sub_turns:
        if not isinstance(item, dict):
            continue
        try:
            a, b = int(item["start_ms"]), int(item["end_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        a, b = max(start, a), min(end, b)
        if b - a < min_segment_ms:
            continue
        spk = item.get("spk")
        label = item.get("spk_label") or item.get("speaker")
        if spk is None and isinstance(label, str) and label.startswith("speaker_"):
            try:
                spk = int(label.split("_", 1)[1])
            except ValueError:
                spk = None
        if not label and spk is not None:
            label = f"speaker_{spk}"
        marks.append((a, b, spk, label))
    if not marks:
        return [utt]
    marks.sort(key=lambda item: (item[0], item[1]))

    segments = []
    cursor = start
    for a, b, spk, label in marks:
        if a - cursor >= min_segment_ms:
            segments.append((cursor, a, main_spk, main_label, False))
        elif a > cursor and segments:
            prev = segments[-1]
            segments[-1] = (prev[0], a, prev[2], prev[3], prev[4])
        seg_start = max(a, cursor)
        if b - seg_start >= min_segment_ms:
            segments.append((seg_start, b, spk, label, True))
        cursor = max(cursor, b)
    if end - cursor >= min_segment_ms:
        segments.append((cursor, end, main_spk, main_label, False))
    elif cursor < end and segments:
        prev = segments[-1]
        segments[-1] = (prev[0], end, prev[2], prev[3], prev[4])

    cleaned = []
    for seg in segments:
        if cleaned and seg[0] < cleaned[-1][1]:
            seg = (cleaned[-1][1], seg[1], seg[2], seg[3], seg[4])
        if seg[1] - seg[0] >= min_segment_ms:
            cleaned.append(seg)

    merged_segments = []
    for seg in cleaned:
        prev = merged_segments[-1] if merged_segments else None
        if (prev is not None and prev[3] == seg[3] and prev[4] == seg[4]
                and seg[0] <= prev[1] + 1):
            merged_segments[-1] = (prev[0], max(prev[1], seg[1]),
                                   prev[2], prev[3], prev[4])
        else:
            merged_segments.append(seg)
    cleaned = merged_segments
    if len(cleaned) <= 1 or not any(seg[4] for seg in cleaned):
        return [utt]

    weights = [seg[1] - seg[0] for seg in cleaned]
    # [D1] raw_text 切分时不调 clean_cjk_spaces，保持原样字符
    raw_parts = _split_text_by_weights(utt.get("raw_text") or "", weights, clean=False)
    polished_parts = _split_text_by_weights(utt.get("polished_text"), weights, clean=True)
    final_parts = _split_text_by_weights(utt.get("final_text"), weights, clean=True)
    parent_id = utt.get("parent_utt_id") or utt["utt_id"]
    out = []
    for index, (seg, raw, polished, final) in enumerate(
            zip(cleaned, raw_parts, polished_parts, final_parts), start=1):
        a, b, spk, label, is_sub = seg
        out.append({
            "utt_id": f"{utt['utt_id']}#{index}",
            "parent_utt_id": parent_id,
            "start_ms": a,
            "end_ms": b,
            "raw_text": raw,
            "polished_text": polished,
            "final_text": final,
            "spk": spk,
            "spk_label": label,
            "overlap_spk": utt.get("overlap_spk"),
            "corrected": bool(utt.get("corrected", False)),
            "bypassed": bool(utt.get("bypassed", False)),
            "error": utt.get("error"),
            "is_sub_turn": is_sub,
        })
    return out


def _join_turn_utterances(group: List[Dict], *, prefer_final: bool = True) -> str:
    """根据句间实际停顿、分组标记与报号边界智能拼接 Turn 文本。"""
    joined = ""
    prev_end_ms: Optional[int] = None
    prev_was_digits = False

    for utt in group:
        final_text = utt.get("final_text") if prefer_final else None
        if final_text is not None:
            raw_piece = final_text
        else:
            raw_piece = utt.get("polished_text") or utt.get("raw_text") or utt.get("text") or ""
        text = clean_cjk_spaces(raw_piece)
        if not text:
            continue

        start_ms = int(utt.get("start_ms") or 0)
        end_ms = int(utt.get("end_ms") or 0)
        is_grouped_cont = bool(utt.get("grouped_cont", False)) if prefer_final else False
        gap_ms = (start_ms - prev_end_ms) if prev_end_ms is not None else 0
        cur_is_digits = _is_pure_spoken_digits(text)

        if joined and joined[-1] not in _END_PUNCT and text[0] not in _END_PUNCT:
            if prev_was_digits and cur_is_digits:
                # [A4] 连续报号子句之间强制加顿号 "、"，防止 "零零六" 与 "零幺三" 粘连成一串
                joined += "、"
            elif _is_cjk(joined[-1]) or _is_cjk(text[0]):
                if not is_grouped_cont and gap_ms >= PAUSE_COMMA_GAP_MS:
                    joined += "，"
            else:
                joined += " "
        joined += text
        prev_end_ms = max(prev_end_ms or 0, end_ms)
        prev_was_digits = cur_is_digits

    return ensure_terminal_punct(joined) if joined else ""


def build_turns(
    utterances: List[Dict],
    *,
    split_gap_ms: int = FINAL_TURN_SPLIT_MS,
    max_turn_duration_ms: int = MAX_TURN_DURATION_MS,
    max_turn_chars: int = MAX_TURN_CHARS,
) -> List[Dict]:
    """把 utterances 合并为 turns (带时长与字数软上限保护 [E8])。"""
    if not utterances:
        return []

    has_labels = any(utt.get("spk_label") is not None for utt in utterances)
    gap_ms = split_gap_ms if has_labels else STREAM_TURN_GAP_MS

    drafts: List[Dict] = []
    for utt in utterances:
        start_ms = int(utt["start_ms"])
        end_ms = int(utt["end_ms"])
        final_text = utt.get("final_text")
        if final_text is not None:
            text = clean_cjk_spaces(final_text)
        else:
            text = clean_cjk_spaces(utt.get("polished_text") or utt.get("raw_text") or "")
        spk_label = utt.get("spk_label")

        if drafts:
            draft = drafts[-1]
            same_spk = (not has_labels) or (spk_label == draft["spk_label"])
            within_gap = (start_ms - draft["end_ms"] < gap_ms)
            projected_dur = max(draft["end_ms"], end_ms) - draft["start_ms"]
            projected_chars = draft["char_count"] + len(text)
            # [E8] 若非同组切分片，且合并后超出 60s 或 500 字软上限，则在句界处自然分段
            not_mid_group = not bool(utt.get("grouped_cont", False))
            within_cap = (
                not not_mid_group
                or (projected_dur <= max_turn_duration_ms and projected_chars <= max_turn_chars)
            )
            if same_spk and within_gap and within_cap:
                draft["end_ms"] = max(draft["end_ms"], end_ms)
                draft["parts"].append(text)
                draft["char_count"] = projected_chars
                draft["utterance_ids"].append(utt["utt_id"])
                draft["group"].append(utt)
                continue

        drafts.append({
            "spk": utt.get("spk"),
            "spk_label": spk_label,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "parts": [text],
            "char_count": len(text),
            "utterance_ids": [utt["utt_id"]],
            "group": [utt],
        })

    turns: List[Dict] = []
    for index, draft in enumerate(drafts, start=1):
        group = draft["group"]
        has_final_asr = any(u.get("final_text") is not None for u in group)
        any_stream_corrected = any(bool(u.get("corrected")) for u in group)
        all_stream_bypassed = bool(group) and all(bool(u.get("bypassed")) for u in group)

        if has_final_asr:
            text_source = "final_asr"
            corrected = False
            bypassed = False
        elif any_stream_corrected:
            text_source = "stream_polished"
            corrected = True
            bypassed = False
        else:
            text_source = "raw"
            corrected = False
            bypassed = all_stream_bypassed

        # [D1] raw_text 原样拼接，不做 clean_cjk_spaces 破坏原貌
        raw_parts = [str(u.get("raw_text") or "").strip() for u in group]
        raw_joined = " ".join([p for p in raw_parts if p])
        stream_joined = _join_turn_utterances(group, prefer_final=False)
        final_joined = _join_turn_utterances(group, prefer_final=True)

        turns.append({
            "turn_id": f"turn_{index:04d}",
            "spk": draft["spk"],
            "spk_label": draft["spk_label"],
            "start_ms": draft["start_ms"],
            "end_ms": draft["end_ms"],
            "text": final_joined,
            "stream_text": stream_joined,
            "raw_text": raw_joined,
            "text_source": text_source,
            "polish_status": "pending",
            "utterance_ids": draft["utterance_ids"],
            "corrected": corrected,
            "bypassed": bypassed,
        })
    return turns


class StreamTurnTracker:
    """流式事件 -> turn 的在线归集器。"""

    def __init__(self, gap_ms: int = STREAM_TURN_GAP_MS) -> None:
        self._lock = threading.Lock()
        self.gap_ms = gap_ms
        self.utt_turn: Dict[str, str] = {}
        self.turn_texts: Dict[str, Dict[str, str]] = {}
        self.last_end_ms: Optional[int] = None
        self.count = 0
        self._current_turn: Optional[str] = None
        self._partial_utt: Optional[str] = None

    def _new_turn(self) -> str:
        self.count += 1
        turn_id = f"turn_{self.count:04d}"
        self.turn_texts[turn_id] = {}
        self._current_turn = turn_id
        return turn_id

    def _turn_for(self, ev: Dict) -> str:
        utt_id = ev.get("utt_id")
        known = self.utt_turn.get(utt_id)
        if known is not None:
            return known

        start_ms = int(ev.get("start_ms") or 0)
        if self._current_turn is None:
            turn_id = self._new_turn()
        elif self.last_end_ms is None:
            turn_id = self._current_turn
        elif start_ms - self.last_end_ms >= self.gap_ms:
            turn_id = self._new_turn()
        else:
            turn_id = self._current_turn

        self.utt_turn[utt_id] = turn_id
        return turn_id

    def _texts(self, turn_id: str) -> List[str]:
        return list(self.turn_texts[turn_id].values())

    def _split_texts(self, turn_id: str):
        partial_utt = self._partial_utt if self.utt_turn.get(self._partial_utt) == turn_id else None
        confirmed: List[str] = []
        partial = ""
        for utt_id, text in self.turn_texts[turn_id].items():
            if utt_id == partial_utt:
                partial = text
            else:
                confirmed.append(text)
        return join_texts(confirmed), partial

    def _annotate(self, ev: Dict, turn_id: str) -> Dict:
        confirmed, partial = self._split_texts(turn_id)
        ev["turn_id"] = turn_id
        ev["turn_confirmed_text"] = confirmed
        ev["turn_partial_text"] = partial
        ev["turn_text"] = join_texts([confirmed, partial]) if partial else confirmed
        return ev

    def assign(self, ev: Dict) -> Dict:
        with self._lock:
            kind = ev.get("t")
            utt_id = ev.get("utt_id")

            if kind == "asr_partial" and utt_id is not None:
                turn_id = self._turn_for(ev)
                self.turn_texts[turn_id][utt_id] = clean_cjk_spaces(ev.get("text") or "")
                self._partial_utt = utt_id
                return self._annotate(ev, turn_id)

            if kind == "asr_final" and utt_id is not None:
                turn_id = self._turn_for(ev)
                self.turn_texts[turn_id][utt_id] = clean_cjk_spaces(ev.get("raw_text") or "")
                if self._partial_utt == utt_id:
                    self._partial_utt = None
                end_ms = ev.get("end_ms")
                if end_ms is not None:
                    self.last_end_ms = max(self.last_end_ms or 0, int(end_ms))
                return self._annotate(ev, turn_id)

            if kind == "llm_confirmed" and utt_id is not None:
                turn_id = self.utt_turn.get(utt_id)
                if turn_id is None:
                    turn_id = self._current_turn
                if turn_id is None:
                    turn_id = self._new_turn()
                self.turn_texts[turn_id][utt_id] = clean_cjk_spaces(ev.get("polished_text") or "")
                ev["turn_polished_text"] = join_texts(self._texts(turn_id))
                return self._annotate(ev, turn_id)

        return ev
