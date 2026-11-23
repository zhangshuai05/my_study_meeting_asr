# -*- coding: utf-8 -*-
"""流式 chunk 档位延迟/原文对拍台。

一次进程内加载模型，逐档跑同一段音频：
- TTFT = 首个 asr_partial 的墙钟时刻 − 该句 start_ms（VAD 起点）
- 定稿延迟 = asr_final 的墙钟时刻 − 该句 end_ms（VAD 终点）
- RTF = 累计推理耗时 / 音频时长（由包住 feed_pcm_chunk 的计时器得出，不是墙钟/音频）

run() 在 fast=False 下按墙钟节流，所以墙钟时刻能直接对齐音频位置。
首档必须先 warmup，否则 torch 初始化会把第一档的 TTFT 整体抬高 200ms 以上。

用法：
    python bench_streaming.py --wav <16k wav> --start 60 --seconds 120 --chunks 300,480,600,960
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

WINDOW_WAV = Path("output/bench/window.wav")
OUT_JSON = Path("output/bench/chunk_sweep.json")


def _cut_window(src, start_s, seconds, dst=WINDOW_WAV):
    import soundfile as sf

    data, _ = sf.read(str(src), dtype="int16", always_2d=True,
                      start=int(16000 * start_s), frames=int(16000 * seconds))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), data[:, 0], 16000, subtype="PCM_16")
    return dst


def _fresh_models():
    """换一份干净模型再测。

    同一份模型连续 generate 会漂：同一段 30s 音频，第 1/2/3 次调用在同一个 token 上
    分别吐出不同的字，其余事件全同；换进程或重载模型即恢复，钉线程数无效。
    不重载，第 N 档比的是"第 N 次调用"而不是"第 N 档参数"。
    """
    from web_server import get_models, unload_models

    unload_models()
    return get_models(need_stream=True, need_offline=False)


def _pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))], 1)


def _timed_factory(factory, cost):
    """包一层计时：run() 内部同步调用 feed_pcm_chunk，这里累计真实推理耗时。"""
    def build(session_wav, models):
        session = factory(session_wav, models)
        original = session.feed_pcm_chunk

        def feed(*a, **kw):
            started = time.monotonic()
            try:
                return original(*a, **kw)
            finally:
                cost["s"] += time.monotonic() - started

        session.feed_pcm_chunk = feed
        return session
    return build


def _run_one(models, wav, chunk, look_back, silence_ms, run):
    from streaming_tuning import make_factory

    events, cost = [], {"s": 0.0}
    started = time.monotonic()
    session_dir = Path("output/sessions") / f"_bench_{chunk[1]}_{silence_ms}"
    obj = run(
        wav, session_dir, chunk_ms=100, fast=False,
        no_llm=True, no_diar=True, no_final_polish=True, no_final_asr=True,
        models=models,
        session_factory=_timed_factory(
            # vad_chunk 传 None，让它跟随 stride；写死 600 会让 300/480 档的
            # VAD 分块与实际喂入长度不一致，测出来的不是那一档。
            make_factory(chunk, look_back, None, silence_ms), cost),
        on_event=lambda ev: events.append((time.monotonic() - started, ev)),
    )
    wall = time.monotonic() - started

    ttft, final_lag, texts, seen = [], [], [], set()
    for t, ev in events:
        if ev.get("t") == "asr_partial" and ev["utt_id"] not in seen:
            seen.add(ev["utt_id"])
            ttft.append((t - ev["start_ms"] / 1000.0) * 1000)
        elif ev.get("t") == "asr_final":
            final_lag.append((t - ev["end_ms"] / 1000.0) * 1000)
            texts.append(ev["raw_text"])

    audio_s = max(1e-9, (obj.get("duration_ms") or 0) / 1000.0)
    return {
        "chunk": list(chunk),
        "chunk_ms": chunk[1] * 60,
        "look_back": list(look_back),
        "silence_ms": silence_ms,
        "utterances": len(texts),
        "ttft_med_ms": _pct(ttft, 0.5),
        "ttft_p90_ms": _pct(ttft, 0.9),
        "final_med_ms": _pct(final_lag, 0.5),
        "final_p90_ms": _pct(final_lag, 0.9),
        "audio_s": round(audio_s, 1),
        "wall_s": round(wall, 1),
        "infer_s": round(cost["s"], 1),
        "rtf": round(cost["s"] / audio_s, 3),
        "chars": sum(len(x) for x in texts),
        "text": "\n".join(texts),
    }


def _edit(a, b):
    """标准 Levenshtein（两行滚动），不引依赖。"""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main(argv=None):
    import bootstrap
    bootstrap.install()
    from run_offline import run
    from streaming_tuning import CHUNK_CHOICES, DEFAULT_CHUNK, make_factory
    from web_server import get_models

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--start", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--chunks", default="300,480,600,960")
    ap.add_argument("--silence", type=int, default=1000)
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--look-back", default="4,1")
    ap.add_argument("--warmup-skip", action="store_true", help="跳过热身（不推荐）")
    args = ap.parse_args(argv)

    look_back = tuple(int(x) for x in args.look_back.split(","))
    by_ms = {v: k for k, v in CHUNK_CHOICES.items()}
    try:
        chunks = [by_ms[f"{int(x)}ms"] for x in args.chunks.split(",")]
    except (KeyError, ValueError):
        raise SystemExit(f"--chunks 只接受 {sorted(by_ms)}，收到 {args.chunks!r}")

    wav = _cut_window(args.wav, args.start, args.seconds)
    models = get_models(need_stream=True, need_offline=False)

    if not args.warmup_skip:
        warm = _cut_window(args.wav, args.start, 6.0, Path("output/bench/warm.wav"))
        t0 = time.monotonic()
        _run_one(models, warm, DEFAULT_CHUNK, look_back, args.silence, run)
        print(f"[warmup] {time.monotonic() - t0:.1f}s（结果丢弃）", flush=True)

    rows = []
    for chunk in chunks:
        # 每档换一份干净模型：同一份模型连续调用会漂（同一段音频在第 1/2/3 次
        # 调用上吐出不同的字），不重载就是在拿"第 N 次调用"跟"第 1 次"比。
        models = _fresh_models()
        row = _run_one(models, wav, chunk, look_back, args.silence, run)
        rows.append(row)
        print(f"[{row['chunk_ms']:4d}ms] ttft_med={row['ttft_med_ms']}ms "
              f"final_med={row['final_med_ms']}ms rtf={row['rtf']} "
              f"utts={row['utterances']} chars={row['chars']}", flush=True)

    base = rows[0]
    for row in rows:
        row["drift_vs_first"] = _edit(row["text"], base["text"])
        print(f"[{row['chunk_ms']:4d}ms] drift={row['drift_vs_first']} "
              f"(字数 {row['chars']} vs {base['chars']})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| chunk | TTFT中位 | TTFT p90 | 定稿中位 | 定稿 p90 | RTF | 句数 | 字数 | 漂移 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['chunk_ms']}ms | {row['ttft_med_ms']} | {row['ttft_p90_ms']} | "
              f"{row['final_med_ms']} | {row['final_p90_ms']} | {row['rtf']} | "
              f"{row['utterances']} | {row['chars']} | {row['drift_vs_first']} |")
    print(f"\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
