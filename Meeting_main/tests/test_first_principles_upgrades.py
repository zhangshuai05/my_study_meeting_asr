# -*- coding: utf-8 -*-
"""第一性原理重构回归测试：
1. SEACO 字级毫秒时间戳精确归位 + 连续切片（消除双垫音回声与按比例腰斩词组）
2. 直通离线转写 `transcribe_audio_offline` 与漏检语音打捞 `recover_uncovered_spans`
3. `run_direct_offline` 跳过流式草稿直出高精度终稿 + 分人 + 精修 + 纪要
4. 真实麦克风实时流式 HTTP 接口 (`/api/live/start` -> `/api/live/chunk` -> `/api/live/finish`)
5. 会后精修并发与双向前后文语境
"""
from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

import bootstrap

bootstrap.install()

import final_polish  # noqa: E402
import offline_asr  # noqa: E402
import run_offline  # noqa: E402
import stream_service  # noqa: E402
from session_store import SessionStore  # noqa: E402


def _make_wav(path: Path, seconds: float = 6.0) -> None:
    sf.write(str(path), np.zeros(int(16000 * seconds), dtype=np.int16), 16000, subtype="PCM_16")


def test_timestamp_alignment_maps_tokens_without_chopping_words(tmp_path: Path):
    wav = tmp_path / "cont.wav"
    _make_wav(wav, seconds=6.0)

    # 模拟 SEACO 在连续段 [0, 4080] 解出 6 个字及其真实毫秒时间戳：
    # 前 2 字落在 u1 [0, 1500]，后 4 字落在 u2 [2000, 4000]
    def fake_seaco(path: str):
        return (
            "小 区 环 境 很 好",
            [[200, 500], [550, 900], [2100, 2400], [2450, 2800], [2900, 3200], [3300, 3700]],
        )

    out = offline_asr.transcribe_utterances(
        wav,
        [
            {"utt_id": "u1", "start_ms": 0, "end_ms": 1500, "raw_text": "一二三四五六"},
            {"utt_id": "u2", "start_ms": 2000, "end_ms": 4000, "raw_text": "七"},
        ],
        asr_call=fake_seaco,
    )

    # 若走旧比例切分（权重 6:1），会把“小区环境很”塞给 u1，把“好”塞给 u2；
    # 时间戳对齐必须精确把“小区”归 u1，“环境很好”归 u2！
    assert out["u1"]["text"] == "小区"
    assert out["u2"]["text"] == "环境很好"
    assert out["u2"]["grouped_cont"] is False


def test_transcribe_audio_offline_splits_by_natural_pause(tmp_path: Path):
    wav = tmp_path / "direct.wav"
    _make_wav(wav, seconds=6.0)

    def fake_seaco(path: str):
        return (
            "大 家 好 下 一 题",
            [[100, 300], [310, 500], [520, 700], [1800, 2000], [2050, 2250], [2300, 2500]],
        )

    utts, dur_ms = offline_asr.transcribe_audio_offline(
        wav,
        asr_call=fake_seaco,
        spans=[(1000, 4000)],
    )

    assert dur_ms == 6000
    assert len(utts) == 2
    assert utts[0]["final_text"] == "大家好"
    assert utts[0]["start_ms"] == 1100 and utts[0]["end_ms"] == 1700
    assert utts[1]["final_text"] == "下一题"
    assert utts[1]["start_ms"] == 2800 and utts[1]["end_ms"] == 3500


def test_recover_uncovered_spans_only_decodes_missing_gaps(tmp_path: Path):
    wav = tmp_path / "rec.wav"
    _make_wav(wav, seconds=10.0)
    calls = []

    def fake_seaco(path: str):
        calls.append(path)
        return ("漏 掉 的 话", [[100, 300], [320, 500], [520, 700], [720, 900]])

    # 真实语音在 [0, 8000]，流式只覆盖了 [0, 3000]，漏掉了 [3080, 8000]
    recovered = offline_asr.recover_uncovered_spans(
        wav,
        [{"utt_id": "utt_0001", "start_ms": 0, "end_ms": 3000, "raw_text": "前半句"}],
        asr_call=fake_seaco,
        spans=[(0, 8000)],
    )

    assert len(calls) == 1
    assert len(recovered) == 1
    assert recovered[0]["utt_id"] == "utt_rec_0001"
    assert recovered[0]["final_text"] == "漏掉的话"
    assert recovered[0]["start_ms"] >= 3000


def test_run_direct_offline_end_to_end(tmp_path: Path):
    wav = tmp_path / "in.wav"
    _make_wav(wav, seconds=3.0)
    sess_dir = tmp_path / "s_direct"

    def fake_direct(audio_path, on_progress=None, should_abort=None):
        return ([
            {"utt_id": "utt_0001", "start_ms": 200, "end_ms": 1800,
             "raw_text": "今天讨论车位问题", "final_text": "今天讨论车位问题"},
        ], 3000)

    obj = run_offline.run_direct_offline(
        wav, sess_dir,
        direct_asr_fn=fake_direct,
        backfill_fn=lambda w, u, m, mode: {"utt_0001": "Speaker_1"},
        final_polish_fn=lambda turns, on_progress=None, should_abort=None: {
            turns[0]["turn_id"]: {
                "text": "今天讨论车位问题。",
                "start_ms": turns[0]["start_ms"],
                "end_ms": turns[0]["end_ms"],
            }
        },
        summary_fn=lambda turns, on_progress=None, should_abort=None: {
            "title": "车位问题讨论",
            "overview": "讨论小区车位问题。",
            "topics": [],
            "decisions": [],
            "action_items": [],
            "speaker_names": {"speaker_0": "主持人"},
            "speaker_roles": {},
            "generated_at": "2026-11-23T18:00:00+08:00",
        },
    )

    assert obj["status"] == "diarized"
    assert obj["turns"][0]["text"] == "今天讨论车位问题。"
    assert obj["summary"]["title"] == "车位问题讨论"
    assert obj["speaker_names"] == {"speaker_0": "主持人"}


def test_final_polish_bidirectional_context_and_parallel_workers():
    turns = [
        {"turn_id": "t1", "spk_label": "speaker_0", "start_ms": 0, "end_ms": 1000, "text": "咱们讨论车尾"},
        {"turn_id": "t2", "spk_label": "speaker_1", "start_ms": 1200, "end_ms": 2500, "text": "对，业主车位确实经常被占"},
    ]
    ctx = final_polish._build_speaker_context(turns, 0)
    assert "后文" in ctx and "业主车位" in ctx

    out = final_polish.polish_turns(
        turns,
        call=lambda p: json.dumps({"polished": "咱们讨论车位。"}, ensure_ascii=False)
        if "车尾" in p.rsplit("待校对整段", 1)[1]
        else json.dumps({"polished": "对，业主车位确实经常被占。"}, ensure_ascii=False),
        max_workers=2,
    )
    assert out["t1"]["text"] == "咱们讨论车位。"
    assert out["t2"]["text"] == "对，业主车位确实经常被占。"


def test_live_mic_http_endpoints_stream_and_finalize(tmp_path: Path, monkeypatch):
    sessions_root = tmp_path / "sessions"

    class _FakeLiveSession:
        def __init__(self, wav_path, models):
            self.wav_path = Path(wav_path)
            self.wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(self.wav_path), np.zeros(1600, dtype=np.int16), 16000, subtype="PCM_16")
            self.completed_utterances = []
            self.processed_ms = 0
            self.chunks = 0

        def feed_pcm_chunk(self, pcm_bytes, is_final=False):
            self.chunks += 1
            if pcm_bytes:
                self.processed_ms += 600
                if self.chunks == 1:
                    return [{"t": "asr_partial", "utt_id": "utt_0001", "start_ms": 0,
                             "current_ms": 600, "text": "大家"}]
                utt = {"utt_id": "utt_0001", "start_ms": 0, "end_ms": 1200, "raw_text": "大家好"}
                self.completed_utterances.append(utt)
                return [{"t": "asr_final", **utt}]
            return []

        def close_wav(self):
            pass

    class _FakePolisher:
        def __init__(self, on_result):
            self.on_result = on_result
            self._res = {}

        def submit(self, ev):
            res = {"polished_text": "大家好。", "corrected": True, "bypassed": False, "error": None}
            self._res[ev["utt_id"]] = res
            self.on_result({**ev, "t": "llm_confirmed", **res})

        def close(self, timeout=None):
            return dict(self._res)

        def abort(self):
            return dict(self._res)

    def _fake_finalize(session_dir, utterances, processed_ms, **kwargs):
        store = SessionStore(Path(session_dir))
        store.write_transcript(utterances, duration_ms=processed_ms)
        store.write_diarization({"utt_0001": {"spk": 0, "spk_label": "speaker_0"}})
        store.write_summary({
            "title": "实时测试会议",
            "overview": "大家打招呼。",
            "topics": [], "decisions": [], "action_items": [],
            "speaker_names": {"speaker_0": "主持人"},
            "speaker_roles": {}, "generated_at": "2026-11-23T18:00:00+08:00",
        })
        return store.build_session_json()

    monkeypatch.setattr(stream_service, "SESSION_ROOT", sessions_root)
    monkeypatch.setattr(stream_service, "_LIVE_SESSION_FACTORY", _FakeLiveSession)
    monkeypatch.setattr(stream_service, "_LIVE_POLISHER_FACTORY", _FakePolisher)
    monkeypatch.setattr(stream_service, "_LIVE_FINALIZE_FN", _fake_finalize)

    server = stream_service.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"

    try:
        # 1. 开始实时录音
        req_start = urllib.request.Request(f"{base}/api/live/start", data=b"", method="POST")
        with urllib.request.urlopen(req_start, timeout=10) as resp:
            start_data = json.loads(resp.read().decode("utf-8"))
        assert start_data["ok"] is True
        sid = start_data["session_id"]

        # 2. 发送两块 PCM，验证实时返回 asr_partial -> asr_final + llm_confirmed
        pcm = b"\x00\x00" * 4800
        req_c1 = urllib.request.Request(
            f"{base}/api/live/chunk?session_id={sid}", data=pcm, method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req_c1, timeout=10) as resp:
            d1 = json.loads(resp.read().decode("utf-8"))
        assert [e["t"] for e in d1["events"]] == ["asr_partial"]

        req_c2 = urllib.request.Request(
            f"{base}/api/live/chunk?session_id={sid}", data=pcm, method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req_c2, timeout=10) as resp:
            d2 = json.loads(resp.read().decode("utf-8"))
        kinds2 = [e["t"] for e in d2["events"]]
        assert "asr_final" in kinds2 and "llm_confirmed" in kinds2

        # 3. 结束录音并流式回传会后结果（分人 + 小秘书纪要 + finish）
        req_fin = urllib.request.Request(
            f"{base}/api/live/finish?session_id={sid}", data=b"", method="POST",
        )
        with urllib.request.urlopen(req_fin, timeout=10) as resp:
            events = [json.loads(line) for line in resp.read().decode("utf-8").splitlines() if line.strip()]
        fin_kinds = [e["t"] for e in events]
        assert "speaker_diarized" in fin_kinds
        assert "meeting_summary" in fin_kinds
        assert fin_kinds[-1] == "finish"
        assert stream_service._SESSION_LOCK.locked() is False
    finally:
        stream_service._abort_live_session_if_active()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
