# -*- coding: utf-8 -*-
"""Session 目录契约：六文件读写 + 原子落盘 + 最终合并。

2026-10-19 审计整改升级：
1. [D1] `utt["raw_text"]` 100% 保持原样，绝不调用 `clean_cjk_spaces` 篡改，与 `utterances.jsonl` 逐字一致。
2. [C1/A5] 准确标记每个 Turn 的 `text_source` 与 `polish_status` (`accepted` | `unchanged` | `rejected`)，
   并保留 `stream_text` 与 `raw_text` 供前端三态视图切换。
3. [D6] 在 `session.json` 中记录 `upstream_baseline` 指纹元数据，确保实验可复现可追溯。
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import bootstrap
from turns import build_turns, clean_cjk_spaces, split_utterance_by_sub_turns

SCHEMA_VERSION = 1

UTTERANCE_KEYS = ("utt_id", "start_ms", "end_ms", "raw_text")

_REPLACE_LOCK = threading.Lock()
_MERGE_LOCK = threading.Lock()
_APPEND_LOCK = threading.Lock()


def new_session_dir(root: Path, when: Optional[datetime] = None) -> Path:
    """创建 output/sessions/s_YYYYMMDD_HHMMSS (重名时追加 _1、_2...)。"""
    when = when or datetime.now()
    base = "s_" + when.strftime("%Y%m%d_%H%M%S")
    root = Path(root)
    candidate = root / base
    suffix = 0
    while True:
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = root / f"{base}_{suffix}"


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    with _REPLACE_LOCK:
        os.replace(tmp, path)


def _append_ndjson(path: Path, obj: Any) -> None:
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)


def _validated_row(utt: Dict[str, Any]) -> Dict[str, Any]:
    missing = [key for key in UTTERANCE_KEYS if key not in utt]
    missing += [key for key in ("utt_id", "start_ms", "end_ms")
                if key in utt and utt[key] is None]
    if missing:
        raise ValueError(f"utterance 缺少字段: {missing}")
    return {key: utt[key] for key in UTTERANCE_KEYS}


class SessionStore:
    """一个 session 目录的读写门面。"""

    def __init__(
        self,
        session_dir: Path,
        *,
        session_id: Optional[str] = None,
        started_at: Optional[str] = None,
    ) -> None:
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or self.dir.name
        self.started_at = started_at or datetime.now().astimezone().isoformat(timespec="seconds")
        self.audio_path = self.dir / "session_audio.wav"
        self._merge_lock = _MERGE_LOCK

    def append_utterance(self, utt: Dict[str, Any]) -> None:
        _append_ndjson(self.dir / "utterances.jsonl", _validated_row(utt))

    def append_event(self, ev: Dict[str, Any]) -> None:
        _append_ndjson(self.dir / "events.ndjson", ev)

    def write_transcript(self, utterances: List[Dict[str, Any]], duration_ms: int) -> Path:
        obj = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "status": "transcript",
            "audio": str(self.audio_path),
            "started_at": self.started_at,
            "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_ms": int(duration_ms),
            "utterances": [_validated_row(utt) for utt in utterances],
        }
        path = self.dir / "transcript.json"
        _atomic_write_json(path, obj)
        return path

    def write_polished(self, mapping: Dict[str, Dict[str, Any]]) -> Path:
        path = self.dir / "polished.json"
        _atomic_write_json(path, mapping)
        return path

    def write_diarization(self, mapping: Dict[str, Dict[str, Any]]) -> Path:
        path = self.dir / "diarization.json"
        _atomic_write_json(path, mapping)
        return path

    def write_turns_polished(self, mapping: Dict[str, Any]) -> Path:
        path = self.dir / "turns_polished.json"
        _atomic_write_json(path, mapping)
        return path

    def write_final_text(self, mapping: Dict[str, Dict]) -> Path:
        path = self.dir / "final_text.json"
        _atomic_write_json(path, mapping)
        return path

    def read_json(self, name: str) -> Optional[Dict[str, Any]]:
        path = self.dir / name
        if not path.exists():
            return None
        with _REPLACE_LOCK:
            return json.loads(path.read_text(encoding="utf-8"))

    def build_session_json(self) -> Dict[str, Any]:
        """读 transcript + 可选的 polished / diarization / final_text / turns_polished，幂等合成 session.json。"""
        with self._merge_lock:
            transcript = self.read_json("transcript.json") or {"utterances": []}
            polished = self.read_json("polished.json") or {}
            diarization = self.read_json("diarization.json")
            final_text = self.read_json("final_text.json") or {}

            merged = []
            for utt in transcript.get("utterances", []):
                uid = utt["utt_id"]
                polish = polished.get(uid, {})
                speaker = (diarization or {}).get(uid, {})
                if not isinstance(speaker, dict):
                    speaker = {}
                ft_entry = final_text.get(uid) or {}
                raw_ft = ft_entry.get("text") if isinstance(ft_entry, dict) else None
                cleaned_ft = clean_cjk_spaces(raw_ft) if raw_ft is not None else None

                # [D1] raw_text 严格保持 transcript.json / utterances.jsonl 中的原始值，不做任何清洗！
                parent = {
                    **utt,
                    "raw_text": utt.get("raw_text", ""),
                    "polished_text": clean_cjk_spaces(
                        polish.get("polished_text", utt.get("raw_text", ""))
                    ),
                    "final_text": cleaned_ft,
                    "grouped_cont": bool(ft_entry.get("grouped_cont", False)) if isinstance(ft_entry, dict) else False,
                    "corrected": bool(polish.get("corrected", False)),
                    "bypassed": bool(polish.get("bypassed", False)),
                    "error": polish.get("error"),
                    "spk": speaker.get("spk"),
                    "spk_label": speaker.get("spk_label"),
                    "overlap_spk": None,
                }
                for part in split_utterance_by_sub_turns(
                        parent, speaker.get("sub_turns")):
                    sub_final = final_text.get(part["utt_id"])
                    if isinstance(sub_final, dict) and sub_final.get("text") is not None:
                        part["final_text"] = clean_cjk_spaces(sub_final["text"])
                        part["grouped_cont"] = bool(sub_final.get("grouped_cont", False))
                    merged.append(part)

            turns = build_turns(merged)
            turns_polished_raw = self.read_json("turns_polished.json")
            has_turns_polished_file = turns_polished_raw is not None
            turns_polished = turns_polished_raw or {}

            for turn in turns:
                entry = turns_polished.get(turn["turn_id"])
                if not isinstance(entry, dict):
                    if has_turns_polished_file:
                        turn["polish_status"] = "rejected"
                    continue
                if (entry.get("start_ms") != turn["start_ms"]
                        or entry.get("end_ms") != turn["end_ms"]):
                    turn["polish_status"] = "mismatch"
                    continue
                polished_text = clean_cjk_spaces(entry.get("text"))
                if isinstance(polished_text, str) and polished_text:
                    actually_changed = polished_text != turn["text"]
                    turn["text"] = polished_text
                    turn["text_source"] = "turn_polished" if actually_changed else turn.get("text_source", "final_asr")
                    turn["polish_status"] = "accepted" if actually_changed else "unchanged"
                    turn["corrected"] = actually_changed

            obj = {
                "schema_version": SCHEMA_VERSION,
                "session_id": self.session_id,
                "status": "diarized" if diarization else "transcript",
                "audio": str(self.audio_path),
                "started_at": transcript.get("started_at", self.started_at),
                "ended_at": transcript.get("ended_at"),
                "duration_ms": transcript.get("duration_ms"),
                "upstream_baseline": {
                    k: {"sha256_16": v[0], "bytes": v[1]}
                    for k, v in getattr(bootstrap, "BASELINE", {}).items()
                },
                "utterances": merged,
                "turns": turns,
            }
            _atomic_write_json(self.dir / "session.json", obj)
            return obj
