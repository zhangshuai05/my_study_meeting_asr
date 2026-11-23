# -*- coding: utf-8 -*-
"""句尾异步 DeepSeek 润色：单消费者 FIFO，连续语境纠错，失败降级。

常规档（面向实时阅读）：
    - 滚动保留最多 800 字高置信度上文 + 句间停顿时长与末尾承接提示，保持人说话的连续性
    - 按读音相近修正同音/近音误识、补自然标点、轻度清理明显口吃叠字
    - 明显未说完的片段不强行加句号；非独立应答的 1~2 字碎片不硬贴句号
    - 严禁编造新内容、改写书面语、颠倒词序或删掉实义词
    - 守卫：长度增幅 > 50% / 缩减 > 30% => 丢弃改写，保留 raw_text

严格档（`strict=True`，测 CER 用）：
    - 只允许原位替换同音字 + 补标点，非标点字数必须完全一致
"""
from __future__ import annotations

import json
import queue
import threading
import unicodedata
import urllib.request
from typing import Callable, Dict, List, Optional

from llm_correct import _get_key, extract_json

MODEL = "deepseek-chat"
API = "https://api.deepseek.com/chat/completions"

SHORT_CIRCUIT_CHARS = 2
MAX_GROWTH = 1.5
MAX_SHRINK = 0.7
CONTEXT_SENTENCES = 3
MAX_CONTEXT_CHARS = 800   # 直播滚动上下文按字符预算截断，不丢掉话题主线
CONTINUITY_GAP_MS = 2500  # 小于此间隔视为紧接上文，很可能属于同一句话的延续
LAST_TAIL_CHARS = 18
PUNCTUATION = "，。！？、,.!? 　"

# 只有明确能独立成句的 1~2 字应答才自动补句号；句中被切出的短碎片（如“一台”“追个”）不硬贴句号
_STANDALONE_SHORT = frozenset({
    "好", "好的", "嗯", "嗯嗯", "对", "对对", "对的", "是的", "是", "行",
    "行吧", "好吧", "可以", "明白", "收到", "谢谢", "没错", "没有", "不是",
    "知道", "哈哈", "哦", "噢", "啊", "哎", "喂", "ok", "OK",
})

SYSTEM_PROMPT = """你是中文会议实时语音转写（ASR）的流式校对助手。
输入是语音识别按停顿切出的实时口语片段，常见问题包括：
1. 同音或近音错字（如平翘舌、前后鼻音、n/l、f/h、声调、连读或口语拆词导致的误识）；
2. 缺少标点，导致长句一口气连在一起难读；
3. 被停顿拦腰切开的半句话或半个词；
4. 明显的口吃叠字或夹在词句中间、打断阅读的无意义“呃”。

你的目标是让实时字幕读起来自然、通顺、符合人说话的连续语意，同时严格遵守：
1. 【按音纠错】结合上文主题与语境，修正明显的同音/近音错字；替换后的字读音必须与原字相同或相近，绝不凭空编造原文没说过的新内容、新事实或总结。人名、专有名词或无法判断处保持原样。
2. 【保持说话连续性】
   - 紧接上文时，注意承接上文末尾可能被切开的半个词或半句话（例如上文末尾是“客”，当前开头是“护的一些反馈”，应修正为“户的一些反馈”），但绝不要重复输出上文已经说过的文字；
   - 绝不要删掉当前片段开头的实义词或动词；
   - 如果当前片段明显没说完（例如停在“这个、那个、然后、但是、把、的、也、在、咱们、提出这个”处，或只是缺少完整主谓结构的名词短语/半截话），末尾不要强行加句号“。”，留空即可；只有语义已经完整结束时，才在末尾加句号、问号或感叹号。
3. 【自然断句与轻度去卡顿】
   - 只使用常见标点（，、。？！），不要插入省略号“……”；
   - 在长片段内部按自然语意和气口补全逗号、顿号、问号；
   - 可清理明显的结巴叠字（如“就就是”“一些些”“个个”“对对”）以及夹在句中明显打断语意的无意义“呃”；
   - 保留说话人原本的口语习惯（如“咱”“是吧”“那”等），不要改写成书面语，不要颠倒词序或删减有效信息。

示范（仅参考校对尺度）：
- 输入：下周三咱们先把测式环境搭起来然后把接扣联调一下看看有没有什么风显再决定要不要延期因为这周大家手头都
  输出：{"polished": "下周三咱们先把测试环境搭起来，然后把接口联调一下，看看有没有什么风险，再决定要不要延期，因为这周大家手头都"}
- 上文末尾：「…目前最大的难题其实是客」（紧接，间隔 0.7 秒）
  输入：护的一些些反馈没法法及时同步过来
  输出：{"polished": "户的一些反馈没法及时同步过来"}
- 输入：咱们再看一下这个
  输出：{"polished": "咱们再看一下这个"}
- 输入：那这个方案大家还有没有什么别的异见没有的话咱们就先这么定
  输出：{"polished": "那这个方案大家还有没有什么别的意见？没有的话咱们就先这么定。"}

只输出严格 JSON 对象，不要解释，不要 markdown 代码块：
{"polished": "校对后的当前片段"}"""

USER_PROMPT_TMPL = """上文语境（仅用于理解主题与同音字，不要续写、不要重复输出）：
{context}

衔接状态：{continuity}

当前语音片段：
{raw}"""

PROMPT_TMPL = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TMPL

# 测 CER 时用严格档：只允许原位替换同音字 + 补标点，非标点字数必须完全一致。
STRICT_PROMPT_TMPL = """你是 ASR 转写纠错器。只允许做一件事：把同音/误识别的字原位替换成正确的字，并补标点。
严禁增字、删字、改词序、改数字、改专有名词、删口语重复、去语气词、改写成书面语。
除标点外，输出字数必须与输入完全一致；不确定就保持原样。

上文语境（仅用于判断同音字，不要续写、不要回应）：
{context}

待纠错句子：
{raw}

输出严格 JSON，不要解释，不要 markdown 代码块：
{{"polished": "纠错后的整句"}}"""


def is_short(raw: str) -> bool:
    """去掉标点与空白后 <= 2 个字符，视为短句。"""
    core = "".join(ch for ch in raw if ch not in PUNCTUATION)
    return len(core) <= SHORT_CIRCUIT_CHARS


def _core(text: str) -> str:
    """去掉所有 Unicode 标点与空白后的正文，用于长度/严格档比较。"""
    return "".join(
        ch for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def accept_rewrite(raw: str, polished: str, *, strict: bool = False) -> bool:
    """守卫：严格档要求非标点字数完全一致；常规档只卡长度增幅/缩减。"""
    candidate = polished.strip()
    if not candidate:
        return False
    if candidate == raw.strip():
        return True
    if strict:
        return len(_core(candidate)) == len(_core(raw))
    if len(candidate) > len(raw) * MAX_GROWTH:
        return False
    if len(candidate) < len(raw) * MAX_SHRINK:
        return False
    return True


def _join_context(segments: List[str]) -> str:
    """把滚动上文自然拼成连续语流，避免短碎片被换行切得支离破碎。"""
    joined = ""
    for seg in segments:
        piece = (seg or "").strip()
        if not piece:
            continue
        if joined and joined[-1] not in "，。！？、；：,.!? ":
            joined += " "
        joined += piece
    return joined


def _build_continuity_hint(
    context_text: str,
    gap_ms: Optional[int],
) -> str:
    """根据句间停顿时长与上文结尾构造连续性提示。"""
    if not context_text:
        return "会议开头或新话题起点"
    last_tail = context_text[-LAST_TAIL_CHARS:]
    if gap_ms is not None and 0 <= gap_ms < CONTINUITY_GAP_MS:
        return (
            f"紧接上一句（仅间隔 {gap_ms / 1000:.1f} 秒），上一句末尾是「{last_tail}」。"
            "很可能是同一句话被切开的后半截，注意自然承接，不要重复输出上一句的字。"
        )
    if gap_ms is not None and gap_ms >= CONTINUITY_GAP_MS:
        return f"与上一句间隔 {gap_ms / 1000:.1f} 秒（有明显停顿，通常是新起一句或换人发言）。"
    return f"紧接上文，上一句末尾是「{last_tail}」。"


def _default_call(
    prompt: str,
    key: Optional[str] = None,
    timeout: float = 60,
    max_tokens: int = 512,
    *,
    system_prompt: Optional[str] = None,
    json_mode: bool = False,
) -> str:
    resolved = _get_key(key)
    if not resolved:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY（进程环境或 Windows 用户级环境变量任一处配置即可）")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    req_body: Dict = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if json_mode:
        req_body["response_format"] = {"type": "json_object"}
    body = json.dumps(req_body).encode("utf-8")
    request = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {resolved}",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


class Polisher:
    """单消费者 FIFO。submit 非阻塞；结果按提交顺序回调 on_result。"""

    def __init__(
        self,
        on_result: Callable[[Dict], None],
        *,
        call: Optional[Callable[[str], str]] = None,
        key: Optional[str] = None,
        timeout: float = 60,
        strict: bool = False,
    ) -> None:
        self._on_result = on_result
        self._timeout = timeout
        self._strict = strict
        self._custom_call = call
        self._key = key
        if call is not None:
            self._call: Callable[[str], str] = call
        else:
            def _bound(prompt: str) -> str:
                return _default_call(prompt, key=key, timeout=timeout, json_mode=True)
            self._call = _bound

        self._queue: "queue.Queue[Optional[Dict]]" = queue.Queue()
        self._results: Dict[str, Dict] = {}
        self._context: List[str] = []
        self._last_end_ms: Optional[int] = None
        self._lock = threading.Lock()
        self._aborted = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="llm-polish", daemon=True)
        self._thread.start()

    # ---- 对外 ----

    def submit(self, ev: Dict) -> None:
        """非阻塞。ev 至少含 utt_id / raw_text / start_ms / end_ms。"""
        if self._aborted.is_set():
            return
        self._queue.put({
            "utt_id": ev["utt_id"],
            "raw_text": ev.get("raw_text") or "",
            "start_ms": ev.get("start_ms"),
            "end_ms": ev.get("end_ms"),
            # 当前 turn 已确认的全部文本；作为首句/重启兜底语境
            "context": ev.get("turn_confirmed_text") or "",
        })

    def results(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._results)

    def close(self, timeout: Optional[float] = None) -> Dict[str, Dict]:
        self._queue.put(None)
        self._thread.join(timeout)
        return self.results()

    def abort(self) -> Dict[str, Dict]:
        """丢弃尚未开始的请求，不等待正在进行的单次 HTTP 调用。"""
        self._aborted.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
        self._queue.put(None)
        return self.results()

    # ---- 内部 ----

    def _invoke_llm(self, user_prompt: str, *, system_prompt: Optional[str] = None) -> str:
        if self._custom_call is not None:
            combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            return self._custom_call(combined)
        return _default_call(
            user_prompt,
            key=self._key,
            timeout=self._timeout,
            system_prompt=system_prompt,
            json_mode=True,
        )

    def _append_context(self, text: str) -> None:
        piece = (text or "").strip()
        if not piece:
            return
        self._context.append(piece)
        # 按字符预算滚动保留，避免几个短碎片就把前面 800 字的主题冲掉
        while len(self._context) > 1 and sum(len(s) for s in self._context) > MAX_CONTEXT_CHARS + 120:
            self._context.pop(0)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                start_ms = item.get("start_ms")
                end_ms = item.get("end_ms")
                gap_ms: Optional[int] = None
                if (isinstance(start_ms, (int, float))
                        and isinstance(self._last_end_ms, (int, float))):
                    diff = int(start_ms) - int(self._last_end_ms)
                    if diff >= 0:
                        gap_ms = diff

                result = self._polish_one(item["raw_text"], item.get("context"), gap_ms=gap_ms)
                if isinstance(end_ms, (int, float)):
                    self._last_end_ms = int(end_ms)
                if self._aborted.is_set():
                    return
                with self._lock:
                    self._results[item["utt_id"]] = result
                # 脏上文传染：只有“无错误、无守卫驳回”的结果才进入后续语境。
                # 失败/被驳回的候选往往是模型幻觉，再喂回去会污染下一句。
                if not result.get("error") and not result.get("rejected_text"):
                    self._append_context(result["polished_text"])
                try:
                    self._on_result({
                        "t": "llm_confirmed",
                        "utt_id": item["utt_id"],
                        "start_ms": item["start_ms"],
                        "end_ms": item["end_ms"],
                        "raw_text": item["raw_text"],
                        "polished_text": result["polished_text"],
                        "corrected": result["corrected"],
                        "bypassed": result["bypassed"],
                        "error": result["error"],
                        "rejected_text": result.get("rejected_text"),
                    })
                except Exception as exc:  # noqa: BLE001 - 回调异常（如落盘失败）不得杀死消费者
                    with self._lock:
                        self._results[item["utt_id"]] = {
                            **result,
                            "error": ((result.get("error") or "") + f" | callback: {type(exc).__name__}: {exc}").strip(" |"),
                        }
            except Exception as exc:  # noqa: BLE001 - 单条任务异常不得杀死消费者
                with self._lock:
                    self._results[item["utt_id"]] = {
                        "polished_text": item.get("raw_text") or "",
                        "corrected": False, "bypassed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            finally:
                self._queue.task_done()

    def _polish_one(
        self,
        raw: str,
        turn_context: Optional[str] = None,
        *,
        gap_ms: Optional[int] = None,
    ) -> Dict:
        base = {
            "polished_text": raw,
            "corrected": False,
            "bypassed": False,
            "error": None,
        }
        if is_short(raw):
            text = raw.strip()
            core = "".join(ch for ch in text if ch not in PUNCTUATION)
            if (core in _STANDALONE_SHORT
                    and text
                    and text[-1] not in "，。！？、,.!?"):
                text += "。"
            return {**base, "polished_text": text, "bypassed": True}

        try:
            # 只取高置信度句子作滚动上文：self._context 里只保存过“无错误、无驳回”的结果。
            # 没有高置信度上文时才退回当前 turn 的原始文本（首句/重启场景）。
            context_text = _join_context(self._context)
            if not context_text:
                context_text = (turn_context or "").strip()
                # asr_final 的 turn 文本末尾就是当前句；留在“上文”里会让模型照抄原句
                if raw and context_text.endswith(raw):
                    context_text = context_text[:-len(raw)].rstrip("，。！？、；：,.!? ")
            context_window = context_text[-MAX_CONTEXT_CHARS:] if context_text else ""
            context = context_window or "（无）"

            if self._strict:
                reply = self._invoke_llm(
                    STRICT_PROMPT_TMPL.format(context=context, raw=raw)
                )
            else:
                continuity = _build_continuity_hint(context_window, gap_ms)
                user_prompt = USER_PROMPT_TMPL.format(
                    context=context,
                    continuity=continuity,
                    raw=raw,
                )
                reply = self._invoke_llm(user_prompt, system_prompt=SYSTEM_PROMPT)

            obj = extract_json(reply)
            polished = obj.get("polished") if isinstance(obj, dict) else None
            if not isinstance(polished, str):
                return {**base, "error": "LLM 输出无法解析为 JSON"}
            candidate = polished.strip()
            if not accept_rewrite(raw, candidate, strict=self._strict):
                return {**base, "error": "守卫驳回改写", "rejected_text": candidate}
            return {
                **base,
                "polished_text": candidate,
                "corrected": candidate != raw.strip(),
            }
        except Exception as exc:  # noqa: BLE001 - 失败必须降级而不是卡死队列
            return {**base, "error": f"{type(exc).__name__}: {exc}"}
