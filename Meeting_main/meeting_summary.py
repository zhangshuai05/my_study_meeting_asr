# -*- coding: utf-8 -*-
"""会后小秘书：把会议定稿发言自动提炼为面向非技术人员的结构化纪要，并推断发言人称呼。

产出 `summary.json`：
    {
      "title": "会议主题",
      "overview": "2~3 句核心摘要",
      "topics": [{"topic": "议题", "points": ["要点1", "要点2"]}],
      "decisions": ["明确结论"],
      "action_items": [{"owner": "负责人", "task": "待办事项", "deadline": "截止时间或空"}],
      "speaker_names": {"speaker_1": "主持人"},
      "speaker_roles": {"speaker_1": "会议主持 / 物业负责人"},
      "generated_at": "2026-11-23T17:10:00+08:00"
    }

设计原则：
1. 忠实原文、绝不硬编：没敲定的结论或待办一律返回空数组，没依据的人名不瞎猜。
2. 自动忽略开场试麦/报号（如“零零六、零幺三、喂喂测试”），不污染主题与摘要。
3. 通用简洁：适合 CEO 助理、产品经理直接复制到工作群或发邮件。
4. 独立 CLI：可对任意历史 session 目录按需补生成 `summary.json`。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import bootstrap

from stream_llm_polish import _default_call
from turns import clean_cjk_spaces

MAX_TRANSCRIPT_CHARS = 30000
MAX_SUMMARY_TOKENS = 2048
DEFAULT_SUMMARY_TIMEOUT = 60.0

SUMMARY_SYSTEM_PROMPT = """你是一位干练的 CEO 助理与资深会议秘书，负责把多人会议转写整理成“简洁、准确、拿来就能发给老板或团队”的结构化会议纪要。

请严格遵守以下规则：
1. 【忠实原文，绝不编造】
   - 所有摘要、结论、待办必须有明确对话依据；
   - 如果整场会只是讨论、提问题或发散，没有明确拍板的结论，`decisions` 就输出空数组 `[]`；
   - 如果没有明确分派的任务，`action_items` 就输出空数组 `[]`；没有提到截止时间时，`deadline` 填 `""`；
   - 自动忽略开场试麦、设备测试或报号（如“喂喂、零零六、零幺三”等），不要把它们写进主题、摘要或结论。
2. 【表达面向非技术人员】
   - 用清晰、自然、简练的中文短句，去掉口语啰嗦与技术术语；
   - `title` 控制在 8～22 字，一眼看出这场会聊啥；
   - `overview` 用 2～3 句话概括会议背景、核心讨论和结果，让没参会的人 10 秒看懂；
   - `topics` 提炼 1～5 个主要议题，每个议题下 2～4 条简明要点。
3. 【发言人称呼与角色推断】
   - 只能针对输入中实际出现的说话人标签（如 `speaker_0`、`speaker_1`）输出；
   - `speaker_names`：仅当对话中明确提到真实姓名、姓氏称呼（如“王总”“李经理”“张帅”）或非常明确的会议身份（如明显组织开会的人填“主持人”）时才填写；完全没有依据就省略该说话人，绝不凭空编造姓名，也不要填回 `speaker_0` 本身；
   - `speaker_roles`：用 4～14 字简短描述这位发言人在会上的主要职责或关注点（例如“会议组织者，引导议题”“反馈小区外来车辆占车位问题”），方便整理人快速辨认是谁。

只输出严格 JSON 对象，不要解释，不要 markdown 代码块，结构如下：
{
  "title": "简短会议主题",
  "overview": "2~3句核心摘要",
  "topics": [
    {"topic": "议题名称", "points": ["要点1", "要点2"]}
  ],
  "decisions": ["明确达成的结论（没有则 []）"],
  "action_items": [
    {"owner": "负责人（优先用推断出的称呼，否则用 speaker_X 或 待定）", "task": "具体事项", "deadline": ""}
  ],
  "speaker_names": {"speaker_1": "主持人"},
  "speaker_roles": {"speaker_1": "组织会议并引导讨论"}
}"""

SUMMARY_USER_TMPL = """以下是本场会议的发言记录（共 {turn_count} 段，涉及说话人：{speakers}）：

{transcript}

请提炼结构化会议纪要并输出严格 JSON。"""


def _format_mmss(ms: Any) -> str:
    try:
        total_s = max(0, int(ms) // 1000)
    except (TypeError, ValueError):
        total_s = 0
    return f"{total_s // 60:02d}:{total_s % 60:02d}"


def default_speaker_display(spk_label: Optional[str]) -> str:
    """把技术标签 speaker_0 转成非技术友好的默认称呼“发言人 1”。"""
    if not spk_label:
        return "发言人"
    label = str(spk_label).strip()
    if label.startswith("speaker_"):
        suffix = label.split("_", 1)[1]
        if suffix.isdigit():
            return f"发言人 {int(suffix) + 1}"
    return label


def build_transcript_text(turns: Sequence[Dict[str, Any]]) -> tuple[str, List[str]]:
    """把 turns 整理为带时间与说话人标签的纯文本；超长时保留首尾避免截断失真。"""
    lines: List[str] = []
    speakers_seen: List[str] = []
    speaker_set = set()

    for turn in turns:
        text = clean_cjk_spaces(turn.get("text") or "")
        if not text:
            continue
        spk = str(turn.get("spk_label") or "speaker_0")
        if spk not in speaker_set:
            speaker_set.add(spk)
            speakers_seen.append(spk)
        ts = _format_mmss(turn.get("start_ms"))
        lines.append(f"[{ts}] {spk}: {text}")

    joined = "\n".join(lines)
    if len(joined) > MAX_TRANSCRIPT_CHARS:
        head_len = int(MAX_TRANSCRIPT_CHARS * 0.65)
        tail_len = MAX_TRANSCRIPT_CHARS - head_len - 40
        joined = (
            joined[:head_len]
            + "\n\n……（中段内容已按长度省略）……\n\n"
            + joined[-tail_len:]
        )
    return joined, speakers_seen


def _clean_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip()


def _is_generic_label(name: str, spk_label: str) -> bool:
    low = name.lower().replace(" ", "")
    if not low or low == spk_label.lower():
        return True
    if low.startswith("speaker_") or low in ("未知", "未知说话人", "待定", "无", "不详"):
        return True
    return False


def normalize_summary(
    obj: Any,
    valid_speakers: Sequence[str],
    *,
    generated_at: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """防御性清洗 LLM 输出，确保字段类型稳定，前端与导出绝不崩溃。"""
    if not isinstance(obj, dict):
        return None

    valid_set = set(valid_speakers)
    title = _clean_str(obj.get("title")) or "会议纪要"
    overview = _clean_str(obj.get("overview"))

    topics: List[Dict[str, Any]] = []
    raw_topics = obj.get("topics")
    if isinstance(raw_topics, list):
        for item in raw_topics[:8]:
            if isinstance(item, dict):
                t_name = _clean_str(item.get("topic") or item.get("title"))
                raw_pts = item.get("points")
                pts = [
                    _clean_str(p)
                    for p in (raw_pts if isinstance(raw_pts, list) else [])
                    if _clean_str(p)
                ]
                if t_name or pts:
                    topics.append({"topic": t_name or "讨论要点", "points": pts[:8]})
            elif isinstance(item, str) and item.strip():
                topics.append({"topic": item.strip(), "points": []})

    decisions: List[str] = []
    raw_dec = obj.get("decisions")
    if isinstance(raw_dec, list):
        for d in raw_dec[:12]:
            s = _clean_str(d)
            if s:
                decisions.append(s)

    action_items: List[Dict[str, str]] = []
    raw_actions = obj.get("action_items")
    if isinstance(raw_actions, list):
        for item in raw_actions[:20]:
            if isinstance(item, dict):
                task = _clean_str(item.get("task") or item.get("content"))
                if not task:
                    continue
                owner = _clean_str(item.get("owner")) or "待定"
                deadline = _clean_str(item.get("deadline"))
                action_items.append({
                    "owner": owner,
                    "task": task,
                    "deadline": deadline,
                })
            elif isinstance(item, str) and item.strip():
                action_items.append({"owner": "待定", "task": item.strip(), "deadline": ""})

    speaker_names: Dict[str, str] = {}
    raw_names = obj.get("speaker_names")
    if isinstance(raw_names, dict):
        for k, v in raw_names.items():
            key = _clean_str(k)
            val = _clean_str(v)
            if (not valid_set or key in valid_set) and not _is_generic_label(val, key):
                speaker_names[key] = val[:24]

    speaker_roles: Dict[str, str] = {}
    raw_roles = obj.get("speaker_roles")
    if isinstance(raw_roles, dict):
        for k, v in raw_roles.items():
            key = _clean_str(k)
            val = _clean_str(v)
            if (not valid_set or key in valid_set) and val:
                speaker_roles[key] = val[:40]

    if not overview and not topics and not decisions and not action_items:
        return None

    return {
        "title": title[:60],
        "overview": overview,
        "topics": topics,
        "decisions": decisions,
        "action_items": action_items,
        "speaker_names": speaker_names,
        "speaker_roles": speaker_roles,
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def summarize_meeting(
    turns: Sequence[Dict[str, Any]],
    *,
    call: Optional[Callable[[str], str]] = None,
    key: Optional[str] = None,
    timeout: float = DEFAULT_SUMMARY_TIMEOUT,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> Optional[Dict[str, Any]]:
    """对完整会议 turns 生成结构化纪要；空内容、中断或调用失败时干净返回 None。"""
    if should_abort is not None and should_abort():
        return None

    transcript, speakers = build_transcript_text(turns)
    if not transcript.strip():
        return None

    if on_progress is not None:
        on_progress({
            "t": "progress",
            "stage": "summary",
            "done": 0,
            "total": 1,
            "msg": "会后小秘书正在整理会议纪要...",
        })

    user_prompt = SUMMARY_USER_TMPL.format(
        turn_count=len(turns),
        speakers="、".join(speakers) if speakers else "未知",
        transcript=transcript,
    )

    try:
        if call is not None:
            reply = call(f"{SUMMARY_SYSTEM_PROMPT}\n\n{user_prompt}")
        else:
            reply = _default_call(
                user_prompt,
                key=key,
                timeout=timeout,
                max_tokens=MAX_SUMMARY_TOKENS,
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                json_mode=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[meeting_summary] 纪要生成跳过: {type(exc).__name__}: {exc}", flush=True)
        return None

    if should_abort is not None and should_abort():
        return None

    from llm_correct import extract_json

    parsed = extract_json(reply)
    summary = normalize_summary(parsed, speakers)

    if on_progress is not None:
        on_progress({
            "t": "progress",
            "stage": "summary",
            "done": 1,
            "total": 1,
            "msg": "会议纪要整理完成" if summary else "会议纪要生成跳过",
        })
    return summary


def format_summary_markdown(
    summary: Optional[Dict[str, Any]],
    *,
    speaker_names: Optional[Dict[str, str]] = None,
    turns: Optional[Sequence[Dict[str, Any]]] = None,
    include_transcript: bool = True,
) -> str:
    """生成适合直接转发微信/飞书/邮件的简洁 Markdown 会议纪要。"""
    names: Dict[str, str] = {}
    if isinstance(summary, dict) and isinstance(summary.get("speaker_names"), dict):
        names.update(summary["speaker_names"])
    if isinstance(speaker_names, dict):
        for k, v in speaker_names.items():
            if v and str(v).strip():
                names[str(k)] = str(v).strip()

    def display_name(label: Optional[str]) -> str:
        if label and label in names:
            return names[label]
        return default_speaker_display(label)

    lines: List[str] = []
    title = (summary.get("title") if isinstance(summary, dict) else None) or "会议纪要"
    lines.append(f"# {title}")
    lines.append("")

    if isinstance(summary, dict):
        overview = _clean_str(summary.get("overview"))
        if overview:
            lines.append(f"> **核心摘要**：{overview}")
            lines.append("")

        decisions = summary.get("decisions") or []
        if decisions:
            lines.append("## 核心结论")
            lines.append("")
            for dec in decisions:
                lines.append(f"- {dec}")
            lines.append("")

        action_items = summary.get("action_items") or []
        if action_items:
            lines.append("## 待办事项")
            lines.append("")
            for item in action_items:
                owner = item.get("owner") or "待定"
                owner_disp = names.get(owner, owner)
                if owner_disp.startswith("speaker_"):
                    owner_disp = default_speaker_display(owner_disp)
                task = item.get("task") or ""
                deadline = _clean_str(item.get("deadline"))
                suffix = f"（截止：{deadline}）" if deadline else ""
                lines.append(f"- [ ] **{owner_disp}**：{task}{suffix}")
            lines.append("")

        topics = summary.get("topics") or []
        if topics:
            lines.append("## 议题要点")
            lines.append("")
            for t in topics:
                lines.append(f"### {t.get('topic') or '讨论要点'}")
                for pt in (t.get("points") or []):
                    lines.append(f"- {pt}")
                lines.append("")

        roles = summary.get("speaker_roles") or {}
        all_spks = list(dict.fromkeys(list(names.keys()) + list(roles.keys())))
        if all_spks:
            lines.append("## 参会发言人")
            lines.append("")
            for spk in all_spks:
                role = _clean_str(roles.get(spk))
                if role:
                    lines.append(f"- **{display_name(spk)}**：{role}")
                else:
                    lines.append(f"- **{display_name(spk)}**")
            lines.append("")

    if include_transcript and turns:
        lines.append("## 完整发言记录")
        lines.append("")
        for turn in turns:
            body = clean_cjk_spaces(turn.get("text") or "")
            if not body:
                continue
            spk = display_name(turn.get("spk_label"))
            ts = _format_mmss(turn.get("start_ms"))
            lines.append(f"- **{spk}**（{ts}）：{body}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="为已有会议目录生成会后小秘书纪要 (summary.json)")
    parser.add_argument("--session", required=True, help="session 目录，如 output/sessions/s_xxx")
    parser.add_argument("--force", action="store_true", help="覆盖已有 summary.json")
    args = parser.parse_args(argv)

    bootstrap.install()
    from session_store import SessionStore

    store = SessionStore(Path(args.session))
    existing = store.read_json("summary.json")
    if existing is not None and not args.force:
        print(f"[meeting_summary] 已有 summary.json，跳过（--force 强制重算）: {store.dir}")
        return 0

    sess = store.read_json("session.json") or store.build_session_json()
    turns = sess.get("turns") or []
    if not turns:
        print(f"[meeting_summary] 会话无有效发言，跳过: {store.dir}", file=sys.stderr)
        return 1

    summary = summarize_meeting(turns)
    if summary is None:
        print("[meeting_summary] 纪要生成失败", file=sys.stderr)
        return 1

    store.write_summary(summary)
    store.build_session_json()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
