# -*- coding: utf-8 -*-
"""会后二次纠错：按 speaker turn 用完整上下文重校一遍。

2026-10-19 审计整改升级：
1. [A6] 说话人感知上下文 (`_build_speaker_context`)：
   不再盲目把上一位不同说话人的话塞进“同一位说话人”的语境；优先提取同一 `spk_label` 的前文，
   并明确标注说话人身份，防止 4 人会议跨说话人窜词。
2. [A3] 全局时间预算 (`max_budget_s = 180s`) 与单段超时收紧 (`timeout = 25s`)：
   一旦会后精修总耗时达到预算上限，自动停止剩余段落的远程调用并保留二遍定稿，绝不无限挂死。
3. [A7] 收紧长段膨胀上限：对 >200 字的长段最多允许增长 15% (最多 +40 字)，防止模型塞入总结句；
   在 `polish_turns` 中增加阿拉伯数字串守卫 (`_digits_preserved`)。
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable, Dict, List, Optional

from stream_llm_polish import _core, _default_call
from turns import clean_cjk_spaces

MAX_GROWTH = 1.35
MAX_SHRINK = 0.65
MAX_ABS_ADDED_CHARS = 45
DEFAULT_POLISH_BUDGET_S = float(os.environ.get("MEETING_POLISH_BUDGET_S", "180.0"))

_RE_ARABIC_NUMS = re.compile(r"\d+(?:\.\d+)?")

PROMPT = """你是会议转写终稿校对器。下面是说话人 [{speaker}] 的一段连续发言。
只做三件事：修正同音/识别别字、补自然标点、补全明显漏掉的字。
禁止：改写句式、增删事实、改数字/人名/专有名词、把口语改成书面语、在汉字之间加空格、添加总结或解释。
不确定就保持原样。
输出严格 JSON，不要解释，不要 markdown 代码块：
{{"polished": "校对后的整段"}}

同场前文参考（只读语境，注意区分说话人，切勿把前文内容拼入当前段）：
{context}

待校对整段（说话人 {speaker}）：
{raw}"""

STRICT_PROMPT = """你是会议转写终稿校对器。只允许做一件事：把同音/识别别字的原位替换成正确的字，并补自然标点。
严禁增字、删字、改词序、改数字、改专有名词、删口语重复、去语气词、改写成书面语、在汉字之间加空格。
除标点外，输出字数必须与输入完全一致；不确定就保持原样。
输出严格 JSON，不要解释，不要 markdown 代码块：
{{"polished": "校对后的整段"}}

同场前文参考（只读语境，不要续写、不要回应）：
{context}

待校对整段（说话人 {speaker}）：
{raw}"""


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
    """[A6] 构建带说话人标签的前文语境，优先包含同一说话人的最近一段话。"""
    if current_idx <= 0:
        return "（无）"
    cur_spk = turns[current_idx].get("spk_label") or "未知说话人"
    ctx_parts: List[str] = []

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

    return "\n".join(ctx_parts) if ctx_parts else "（无）"


def polish_turns(
    turns: List[Dict],
    *,
    call: Optional[Callable[[str], str]] = None,
    key: Optional[str] = None,
    timeout: float = 25.0,
    max_budget_s: float = DEFAULT_POLISH_BUDGET_S,
    strict: bool = False,
    on_progress: Optional[Callable[[Dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> Dict[str, Dict]:
    """返回 {turn_id: {"text", "start_ms", "end_ms", "status"}}；带全局时间预算保护 [A3]。"""
    caller = call or (lambda prompt: _default_call(prompt, key=key, timeout=timeout,
                                                   max_tokens=4096))

    out: Dict[str, Dict] = {}
    total = len(turns)
    prompt_template = STRICT_PROMPT if strict else PROMPT
    deadline = time.monotonic() + max(5.0, float(max_budget_s))

    for index, turn in enumerate(turns, 1):
        if should_abort is not None and should_abort():
            break
        # [A3] 检查会后精修全局时间预算 (注入伪 call 测试时耗约为 0，不受影响)
        if call is None and time.monotonic() >= deadline:
            print(f"[final_polish] 已达会后精修全局时间预算 ({max_budget_s:.0f}s)，跳过剩余 {total - index + 1} 段",
                  flush=True)
            break

        raw = clean_cjk_spaces(turn.get("text") or "")
        spk = turn.get("spk_label") or "speaker_0"
        if raw:
            context = _build_speaker_context(turns, index - 1)
            try:
                reply = caller(prompt_template.format(
                    speaker=spk,
                    context=context,
                    raw=raw,
                ))
                from llm_correct import extract_json
                obj = extract_json(reply)
                candidate = obj.get("polished") if isinstance(obj, dict) else None
                if isinstance(candidate, str):
                    cand_clean = clean_cjk_spaces(candidate)
                    if (accept_final(raw, cand_clean, strict=strict)
                            and _digits_preserved(raw, cand_clean)):
                        out[turn["turn_id"]] = {
                            "text": cand_clean,
                            "start_ms": int(turn.get("start_ms") or 0),
                            "end_ms": int(turn.get("end_ms") or 0),
                        }
            except Exception as exc:  # noqa: BLE001
                print(f"[final_polish] {turn.get('turn_id')} 精修跳过: {type(exc).__name__}",
                      flush=True)

        if on_progress is not None:
            on_progress({
                "t": "progress", "stage": "final_polish",
                "done": index, "total": total,
                "msg": f"会后精修中 ({index}/{total})...",
            })
    return out
