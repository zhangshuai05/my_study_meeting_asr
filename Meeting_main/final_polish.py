# -*- coding: utf-8 -*-
"""会后二次纠错：按 speaker turn 用前后完整上下文并发重校。

升级要点：
1. [A6+] 双向说话人感知上下文 (`_build_speaker_context`)：
   同时提供同一说话人的稍早发言、紧邻前一句以及下一句后文参考，大幅提升跨句同音字消歧。
2. [并发提速 + Prefix Cache] 使用 `system_prompt` + `json_mode=True` + 4 路并发线程池：
   命中 DeepSeek 前缀缓存、消除 JSON 格式畸变，长会议精修耗时缩减 ~4 倍，不再半途触达预算上限。
3. [A3/A7] 全局时间预算 (`max_budget_s = 180s`)、长段膨胀上限与阿拉伯数字串守卫 (`_digits_preserved`)。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

from stream_llm_polish import _core, _default_call
from turns import clean_cjk_spaces

MAX_GROWTH = 1.35
MAX_SHRINK = 0.65
MAX_ABS_ADDED_CHARS = 45
DEFAULT_POLISH_BUDGET_S = float(os.environ.get("MEETING_POLISH_BUDGET_S", "180.0"))
DEFAULT_POLISH_WORKERS = int(os.environ.get("MEETING_POLISH_WORKERS", "4"))

_RE_ARABIC_NUMS = re.compile(r"\d+(?:\.\d+)?")

SYSTEM_PROMPT = """你是会议转写终稿校对器。输入是某位说话人的一段连续发言。
只做三件事：按发音相近修正同音/近音识别别字、补全自然标点、补全明显吞掉的虚字。
严格禁止：改写句式、增删事实、篡改数字/人名/专有名词、把口语改成书面语、在汉字之间加空格、添加总结或解释、把前后文参考拼进当前段。
不确定就保持原样。
只输出严格 JSON 对象，不要解释，不要 markdown 代码块：
{"polished": "校对后的整段"}"""

USER_PROMPT = """同场前后文参考（只读语境，注意区分说话人，切勿把参考内容拼入当前段）：
{context}

待校对整段（说话人 {speaker}）：
{raw}"""

PROMPT = SYSTEM_PROMPT + "\n\n" + USER_PROMPT

STRICT_SYSTEM_PROMPT = """你是会议转写终稿校对器。只允许做一件事：把同音/识别别字原位替换成正确的字，并补自然标点。
严禁增字、删字、改词序、改数字、改专有名词、删口语重复、去语气词、改写成书面语、在汉字之间加空格。
除标点外，输出字数必须与输入完全一致；不确定就保持原样。
只输出严格 JSON 对象，不要解释，不要 markdown 代码块：
{"polished": "校对后的整段"}"""

STRICT_USER_PROMPT = """同场前后文参考（只读语境，不要续写、不要回应）：
{context}

待校对整段（说话人 {speaker}）：
{raw}"""

STRICT_PROMPT = STRICT_SYSTEM_PROMPT + "\n\n" + STRICT_USER_PROMPT


def _digits_preserved(raw: str, candidate: str) -> bool:
    """检查原文中的阿拉伯数字序列是否被模型篡改 (如把 106 改成 108)。"""
    raw_nums = _RE_ARABIC_NUMS.findall(raw)
    if not raw_nums:
        return True
    cand_nums = _RE_ARABIC_NUMS.findall(candidate)
    return raw_nums == cand_nums


def accept_final(raw: str, candidate: str, *, strict: bool = False) -> bool:
    """会后档守卫：基于去除汉字间多余空格后的真实文本长度卡 1.35/0.65。"""
    text = clean_cjk_spaces(candidate)
    raw_clean = clean_cjk_spaces(raw)
    if not text:
        return False
    if text == raw_clean:
        return True
    if strict:
        return len(_core(text)) == len(_core(raw_clean))
    if len(text) > max(1, len(raw_clean)) * MAX_GROWTH:
        return False
    # [A7] 防止 900 字长段合法多出 300 字总结：非标点净增字数不得超过 MAX_ABS_ADDED_CHARS (45字)
    if len(raw_clean) >= 120 and (len(_core(text)) - len(_core(raw_clean))) > MAX_ABS_ADDED_CHARS:
        return False
    if len(text) < len(raw_clean) * MAX_SHRINK:
        return False
    return True


def _build_speaker_context(turns: List[Dict], current_idx: int) -> str:
    """[A6+] 构建带说话人标签的前后文语境：优先同说话人稍早发言 + 紧邻前一句 + 紧邻后一句。"""
    cur_spk = turns[current_idx].get("spk_label") or "未知说话人"
    ctx_parts: List[str] = []

    if current_idx > 0:
        # 1. 查找同一说话人此前最近的一段发言
        for j in range(current_idx - 1, max(-1, current_idx - 6), -1):
            prev_spk = turns[j].get("spk_label") or "未知说话人"
            if prev_spk == cur_spk and j != current_idx - 1:
                prev_txt = clean_cjk_spaces(turns[j].get("text") or "")[-160:]
                if prev_txt:
                    ctx_parts.append(f"[{cur_spk} 稍早发言]: {prev_txt}")
                break

        # 2. 紧邻的上一句发言 (标明说话人，防止跨人串话)
        imm_prev = turns[current_idx - 1]
        imm_spk = imm_prev.get("spk_label") or "未知说话人"
        imm_txt = clean_cjk_spaces(imm_prev.get("text") or "")[-200:]
        if imm_txt:
            tag = "同说话人上一句" if imm_spk == cur_spk else f"另一位说话人 {imm_spk}"
            ctx_parts.append(f"[{tag}]: {imm_txt}")
    else:
        ctx_parts.append("（无）")

    # 3. 紧邻的下一句后文（仅在存在后续发言时补充，用于判断当前句尾同音字）
    if current_idx + 1 < len(turns):
        nxt = turns[current_idx + 1]
        nxt_spk = nxt.get("spk_label") or "未知说话人"
        nxt_txt = clean_cjk_spaces(nxt.get("text") or "")[:140]
        if nxt_txt:
            tag = "同说话人下一句" if nxt_spk == cur_spk else f"随后 {nxt_spk} 接话"
            ctx_parts.append(f"[后文·{tag}]: {nxt_txt}")

    return "\n".join(ctx_parts) if ctx_parts else "（无）"


def polish_turns(
    turns: List[Dict],
    *,
    call: Optional[Callable[[str], str]] = None,
    key: Optional[str] = None,
    timeout: float = 25.0,
    max_budget_s: float = DEFAULT_POLISH_BUDGET_S,
    strict: bool = False,
    max_workers: Optional[int] = None,
    on_progress: Optional[Callable[[Dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> Dict[str, Dict]:
    """返回 {turn_id: {"text", "start_ms", "end_ms"}}；支持并发加速与全局时间预算保护。"""
    out: Dict[str, Dict] = {}
    total = len(turns)
    if total == 0:
        return out

    sys_prompt = STRICT_SYSTEM_PROMPT if strict else SYSTEM_PROMPT
    usr_template = STRICT_USER_PROMPT if strict else USER_PROMPT
    deadline = time.monotonic() + max(5.0, float(max_budget_s))

    # 注入测试伪 call 时保持串行确定性；真实远程调用默认 4 路并发提速
    workers = max_workers if max_workers is not None else (1 if call is not None else DEFAULT_POLISH_WORKERS)
    workers = max(1, min(int(workers), total))

    def _polish_one(idx: int) -> Tuple[str, Optional[Dict]]:
        turn = turns[idx]
        tid = turn["turn_id"]
        if should_abort is not None and should_abort():
            return tid, None
        if call is None and time.monotonic() >= deadline:
            return tid, None

        raw = clean_cjk_spaces(turn.get("text") or "")
        if not raw:
            return tid, None

        spk = turn.get("spk_label") or "speaker_0"
        context = _build_speaker_context(turns, idx)
        user_msg = usr_template.format(speaker=spk, context=context, raw=raw)
        try:
            if call is not None:
                reply = call(f"{sys_prompt}\n\n{user_msg}")
            else:
                reply = _default_call(
                    user_msg,
                    key=key,
                    timeout=timeout,
                    max_tokens=4096,
                    system_prompt=sys_prompt,
                    json_mode=True,
                )
            from llm_correct import extract_json
            obj = extract_json(reply)
            candidate = obj.get("polished") if isinstance(obj, dict) else None
            if isinstance(candidate, str):
                cand_clean = clean_cjk_spaces(candidate)
                if (accept_final(raw, cand_clean, strict=strict)
                        and _digits_preserved(raw, cand_clean)):
                    return tid, {
                        "text": cand_clean,
                        "start_ms": int(turn.get("start_ms") or 0),
                        "end_ms": int(turn.get("end_ms") or 0),
                    }
        except Exception as exc:  # noqa: BLE001
            print(f"[final_polish] {tid} 精修跳过: {type(exc).__name__}", flush=True)
        return tid, None

    if workers <= 1:
        for index in range(total):
            if should_abort is not None and should_abort():
                break
            if call is None and time.monotonic() >= deadline:
                print(
                    f"[final_polish] 已达会后精修全局时间预算 ({max_budget_s:.0f}s)，跳过剩余 {total - index} 段",
                    flush=True,
                )
                break
            tid, entry = _polish_one(index)
            if entry is not None:
                out[tid] = entry
            if on_progress is not None:
                on_progress({
                    "t": "progress", "stage": "final_polish",
                    "done": index + 1, "total": total,
                    "msg": f"会后精修中 ({index + 1}/{total})...",
                })
        return out

    done_count = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="final-polish") as pool:
        futures = []
        for idx in range(total):
            if should_abort is not None and should_abort():
                break
            futures.append(pool.submit(_polish_one, idx))
        for fut in as_completed(futures):
            tid, entry = fut.result()
            if entry is not None:
                out[tid] = entry
            done_count += 1
            if on_progress is not None:
                on_progress({
                    "t": "progress", "stage": "final_polish",
                    "done": done_count, "total": total,
                    "msg": f"会后精修中 ({done_count}/{total})...",
                })
    return out
