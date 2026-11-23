# -*- coding: utf-8 -*-
"""会后小秘书单元测试：摘要清洗、Markdown 导出、SessionStore 持久化、run_offline 集成与 HTTP 接口。"""
from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import bootstrap

bootstrap.install()

import run_offline  # noqa: E402
import stream_service  # noqa: E402
from meeting_summary import (  # noqa: E402
    default_speaker_display,
    format_summary_markdown,
    normalize_summary,
    summarize_meeting,
)
from session_store import SessionStore  # noqa: E402


SAMPLE_TURNS = [
    {
        "turn_id": "turn_0001",
        "spk": 0,
        "spk_label": "speaker_0",
        "start_ms": 0,
        "end_ms": 12000,
        "text": "今天把各位叫过来，主要讨论小区外来车辆占车位的问题。",
    },
    {
        "turn_id": "turn_0002",
        "spk": 1,
        "spk_label": "speaker_1",
        "start_ms": 13000,
        "end_ms": 28000,
        "text": "王经理说得对，我周五前把门禁登记方案拉出来给大家评审。",
    },
]

SAMPLE_LLM_JSON = {
    "title": "小区外来车辆与车位管理讨论",
    "overview": "会议围绕小区外来车辆占用业主车位问题展开讨论，明确由发言人 2 在周五前输出门禁登记方案。",
    "topics": [
        {
            "topic": "外来车辆占车位治理",
            "points": ["业主反馈外来车辆随意停放导致纠纷", "需加强门禁与登记管理"],
        }
    ],
    "decisions": ["先制定门禁登记方案再组织评审"],
    "action_items": [
        {"owner": "speaker_1", "task": "输出门禁登记方案并提交评审", "deadline": "本周五前"}
    ],
    "speaker_names": {
        "speaker_0": "王经理",
        "speaker_1": "speaker_1",
        "speaker_99": "不存在的人",
    },
    "speaker_roles": {
        "speaker_0": "会议主持，提出车辆问题",
        "speaker_1": "负责落地门禁方案",
        "speaker_99": "应被过滤",
    },
}


def test_default_speaker_display():
    assert default_speaker_display("speaker_0") == "发言人 1"
    assert default_speaker_display("speaker_2") == "发言人 3"
    assert default_speaker_display(None) == "发言人"


def test_normalize_summary_filters_invalid_and_generic_speakers():
    out = normalize_summary(SAMPLE_LLM_JSON, ["speaker_0", "speaker_1"])
    assert out is not None
    assert out["title"] == "小区外来车辆与车位管理讨论"
    assert out["speaker_names"] == {"speaker_0": "王经理"}
    assert set(out["speaker_roles"]) == {"speaker_0", "speaker_1"}
    assert len(out["action_items"]) == 1
    assert out["generated_at"]


def test_normalize_summary_rejects_empty_payload():
    assert normalize_summary({}, ["speaker_0"]) is None
    assert normalize_summary("not a dict", ["speaker_0"]) is None


def test_summarize_meeting_emits_progress_and_returns_summary():
    progress = []
    prompts = []

    def fake_call(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(SAMPLE_LLM_JSON, ensure_ascii=False)

    summary = summarize_meeting(SAMPLE_TURNS, call=fake_call, on_progress=progress.append)
    assert summary is not None
    assert summary["title"] == "小区外来车辆与车位管理讨论"
    assert "speaker_0" in prompts[0] and "小区外来车辆" in prompts[0]
    assert [p["done"] for p in progress] == [0, 1]
    assert all(p["stage"] == "summary" for p in progress)


def test_summarize_meeting_respects_abort_and_fails_soft():
    called = []
    assert summarize_meeting(
        SAMPLE_TURNS,
        call=lambda p: called.append(p) or "{}",
        should_abort=lambda: True,
    ) is None
    assert called == []

    def boom(prompt: str) -> str:
        raise RuntimeError("network down")

    assert summarize_meeting(SAMPLE_TURNS, call=boom) is None
    assert summarize_meeting(SAMPLE_TURNS, call=lambda p: "not json") is None


def test_format_summary_markdown_uses_manual_speaker_names():
    summary = normalize_summary(SAMPLE_LLM_JSON, ["speaker_0", "speaker_1"])
    md = format_summary_markdown(
        summary,
        speaker_names={"speaker_0": "王总", "speaker_1": "李工"},
        turns=SAMPLE_TURNS,
    )
    assert "# 小区外来车辆与车位管理讨论" in md
    assert "**李工**：输出门禁登记方案并提交评审（截止：本周五前）" in md
    assert "**王总**（00:00）" in md
    assert "turn_0001" not in md


def test_session_store_merges_summary_and_manual_speaker_names(tmp_path: Path):
    store = SessionStore(tmp_path / "s_20261123_170000")
    store.write_transcript([
        {"utt_id": "utt_0001", "start_ms": 0, "end_ms": 1000, "raw_text": "大家好"},
    ], duration_ms=1000)
    store.write_diarization({"utt_0001": {"spk": 0, "spk_label": "speaker_0"}})

    summary = normalize_summary(SAMPLE_LLM_JSON, ["speaker_0", "speaker_1"])
    store.write_summary(summary)
    obj = store.build_session_json()
    assert obj["summary"]["title"] == "小区外来车辆与车位管理讨论"
    assert obj["speaker_names"] == {"speaker_0": "王经理"}

    # 手工改名优先覆盖 AI 推断
    store.write_speaker_names({"speaker_0": "王总"})
    obj2 = store.build_session_json()
    assert obj2["speaker_names"]["speaker_0"] == "王总"


def test_run_offline_runs_summary_and_handles_failure(tmp_path: Path):
    wav = tmp_path / "in.wav"
    sf.write(str(wav), np.zeros(3200, dtype=np.int16), 16000, subtype="PCM_16")

    class _FakeSession:
        def __init__(self, wav_path):
            Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
            Path(wav_path).write_bytes(b"RIFF")
            self.completed_utterances = []
            self.processed_ms = 0

        def feed_pcm_chunk(self, pcm_bytes, is_final=False):
            if is_final:
                utt = {"utt_id": "utt_0001", "start_ms": 0, "end_ms": 200, "raw_text": "我们下周五上线"}
                self.completed_utterances.append(utt)
                self.processed_ms = 200
                return [{"t": "asr_final", **utt}]
            return []

    class _FakePolisher:
        def __init__(self, on_result):
            self.on_result = on_result

        def submit(self, ev):
            self.on_result({
                "t": "llm_confirmed", **ev,
                "polished_text": "我们下周五上线。", "corrected": True, "bypassed": False, "error": None,
            })

        def close(self, timeout=None):
            return {"utt_0001": {"polished_text": "我们下周五上线。", "corrected": True,
                                 "bypassed": False, "error": None}}

    summary_out = normalize_summary(SAMPLE_LLM_JSON, ["speaker_0"])
    sess_dir = tmp_path / "s_ok"
    obj = run_offline.run(
        wav, sess_dir, chunk_ms=200, fast=True, no_final_asr=True, no_final_polish=True,
        session_factory=lambda p, m: _FakeSession(p),
        polisher_factory=_FakePolisher,
        backfill_fn=lambda w, u, m, mode: {"utt_0001": "Speaker_1"},
        summary_fn=lambda turns, on_progress=None, should_abort=None: summary_out,
        models={},
    )
    assert (sess_dir / "summary.json").exists()
    assert obj["summary"]["title"] == "小区外来车辆与车位管理讨论"
    assert obj["speaker_names"] == {"speaker_0": "王经理"}

    # summary 抛异常绝不破坏会议结果
    sess_fail = tmp_path / "s_fail"
    obj_fail = run_offline.run(
        wav, sess_fail, chunk_ms=200, fast=True, no_final_asr=True, no_final_polish=True,
        session_factory=lambda p, m: _FakeSession(p),
        polisher_factory=_FakePolisher,
        backfill_fn=lambda w, u, m, mode: {"utt_0001": "Speaker_1"},
        summary_fn=lambda turns, on_progress=None, should_abort=None: (_ for _ in ()).throw(RuntimeError("api fail")),
        models={},
    )
    assert obj_fail["summary"] is None
    assert obj_fail["turns"][0]["text"] == "我们下周五上线。"


def test_http_summary_and_speaker_rename_endpoints(tmp_path: Path, monkeypatch):
    sessions_root = tmp_path / "sessions"
    sid = "s_20261123_170500"
    store = SessionStore(sessions_root / sid)
    store.write_transcript([
        {"utt_id": "utt_0001", "start_ms": 0, "end_ms": 2000, "raw_text": "今天讨论车位管理"},
    ], duration_ms=2000)
    store.write_diarization({"utt_0001": {"spk": 0, "spk_label": "speaker_0"}})
    store.build_session_json()

    expected_summary = normalize_summary(SAMPLE_LLM_JSON, ["speaker_0"])
    monkeypatch.setattr(stream_service, "SESSION_ROOT", sessions_root)
    monkeypatch.setattr(stream_service, "_SUMMARY_FN", lambda turns: expected_summary)

    server = stream_service.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"

    try:
        # 1. 按需生成会议纪要
        req = urllib.request.Request(f"{base}/api/sessions/{sid}/summary", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
        assert data["ok"] is True
        assert data["summary"]["title"] == "小区外来车辆与车位管理讨论"
        assert data["speaker_names"] == {"speaker_0": "王经理"}

        # 2. 历史列表带上标题与 has_summary
        with urllib.request.urlopen(f"{base}/api/sessions", timeout=10) as resp:
            listing = json.loads(resp.read().decode("utf-8"))
        assert listing["sessions"][0]["title"] == "小区外来车辆与车位管理讨论"
        assert listing["sessions"][0]["has_summary"] is True

        # 3. 手工修改发言人姓名并持久化
        body = json.dumps({"speaker_names": {"speaker_0": "张总"}}, ensure_ascii=False).encode("utf-8")
        req2 = urllib.request.Request(
            f"{base}/api/sessions/{sid}/speakers",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req2, timeout=10) as resp:
            assert resp.status == 200
            renamed = json.loads(resp.read().decode("utf-8"))
        assert renamed["speaker_names"]["speaker_0"] == "张总"

        # 4. 跨站 Origin 必须被拒绝
        bad_req = urllib.request.Request(
            f"{base}/api/sessions/{sid}/speakers",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Origin": "https://evil.example.com"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad_req, timeout=10)
        assert exc.value.code == 403
        exc.value.close()

        # 5. 音频回放接口支持完整返回与 Range 206 跳转
        sf.write(str(sessions_root / sid / "session_audio.wav"),
                 np.zeros(1600, dtype=np.int16), 16000, subtype="PCM_16")
        with urllib.request.urlopen(f"{base}/api/sessions/{sid}/audio", timeout=10) as resp:
            assert resp.status == 200
            assert resp.headers.get("Accept-Ranges") == "bytes"
            full_bytes = resp.read()
            assert full_bytes.startswith(b"RIFF")

        range_req = urllib.request.Request(
            f"{base}/api/sessions/{sid}/audio",
            headers={"Range": "bytes=0-15"},
        )
        with urllib.request.urlopen(range_req, timeout=10) as resp:
            assert resp.status == 206
            assert resp.headers.get("Content-Range", "").startswith("bytes 0-15/")
            assert resp.read() == full_bytes[:16]

        # 6. 删除历史会话（跨站拒绝、运行中 409、正常删除 200）
        del_bad = urllib.request.Request(
            f"{base}/api/sessions/{sid}",
            method="DELETE",
            headers={"Origin": "https://evil.example.com"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(del_bad, timeout=10)
        assert exc.value.code == 403
        exc.value.close()

        del_req = urllib.request.Request(f"{base}/api/sessions/{sid}", method="DELETE")
        with urllib.request.urlopen(del_req, timeout=10) as resp:
            assert resp.status == 200
            assert json.loads(resp.read().decode("utf-8"))["ok"] is True
        assert not (sessions_root / sid).exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
