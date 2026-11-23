# -*- coding: utf-8 -*-
"""默认参数等价性：裸 StreamingSession 与 make_factory(默认) 必须逐事件一致。

这是「参数化没有改变默认行为」的唯一观测证据，不是论证。

三条纪律：

1. 只比对由样本下标决定的东西 —— (type, text, start_ms, end_ms/current_ms)。
   不比墙钟：热身抖动会让夹具自己红，而那不是行为差异。

2. 每一边都必须是「模型上的第一次调用」。原因不是洁癖：
   实测同一份模型连续调用会漂 —— 同一段 30s 音频，第 1/2/3 次调用在 #15 号事件
   上分别吐出「尽天啊把 / 今天啊把 / 尽量啊把」，其余 25 条事件完全相同；
   换进程或在两次调用之间重载模型都恢复一致，`torch.set_num_threads(1)` 无效。
   也就是说这是模型对象自带的跨调用状态，不是并行浮点噪声。
   不重载就比，比的是"第 1 次 vs 第 2 次"，而不是"raw vs tuned"。
   （该泄漏本身是独立缺陷，见 reports；不在此夹具里顺手修。）

3. 已知瑕疵（首句 start_ms=0 的「喂你」、utt_0438/0439 重叠 150ms）不做归一化。
   它们应当原样同时出现在两边；在这个夹具里修掉，等于把回归钉和行为改动混进一次提交。
"""
from pathlib import Path

import pytest

import bootstrap

bootstrap.install()

import soundfile as sf  # noqa: E402

SMOKE = Path(r"D:\Deepseek\yinp\smoke_clip_180s.wav")
SECONDS = 30.0
CHUNK = 9600  # 600ms @16kHz，与上游默认 stride 相同


def _collect(session, audio):
    events = []
    for start in range(0, len(audio), CHUNK):
        piece = audio[start:start + CHUNK]
        events.extend(session.feed_pcm_chunk(
            piece.tobytes(), is_final=(start + CHUNK) >= len(audio)))
    return events


def _shape(events):
    """只保留样本下标决定的内容，丢掉任何与墙钟/耗时相关的字段。"""
    shaped = []
    for ev in events:
        if ev["t"] == "asr_partial":
            shaped.append(("asr_partial", ev["text"], ev["start_ms"], ev["current_ms"]))
        elif ev["t"] == "asr_final":
            shaped.append(("asr_final", ev["raw_text"], ev["start_ms"], ev["end_ms"]))
        else:
            shaped.append((ev["t"],))
    return shaped


def _first_diff(left, right):
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return f"#{idx}\n  raw  ={a!r}\n  tuned={b!r}"
    if len(left) != len(right):
        return f"长度不同 {len(left)} vs {len(right)}"
    return "<identical>"


def _fresh_models():
    from web_server import get_models, unload_models

    unload_models()  # 丢掉上一个 session 用过的模型，保证这是新对象的第一次调用
    return get_models(need_stream=True, need_offline=False)


@pytest.mark.slow
@pytest.mark.skipif(not SMOKE.exists(), reason="缺少 smoke_clip_180s.wav")
def test_default_params_match_upstream(tmp_path):
    from streaming_asr_engine import StreamingSession
    from streaming_tuning import make_factory

    data, _ = sf.read(str(SMOKE), dtype="int16", always_2d=True,
                      frames=int(16000 * SECONDS))
    audio = data[:, 0]

    # 左：上游默认构造
    models = _fresh_models()
    left = _shape(_collect(
        StreamingSession(str(tmp_path / "raw.wav"),
                         asr_model=models["stream_asr"],
                         vad_model=models["vad"],
                         max_end_silence_time=1000),
        audio))

    # 右：make_factory() 默认参数（先构造、再 tune，与 bench 走同一条注入路径）
    models = _fresh_models()
    right = _shape(_collect(make_factory()(str(tmp_path / "tuned.wav"), models), audio))

    # 防空转：两边都没检出语音时相等是真空通过。
    finals = [x for x in left if x[0] == "asr_final"]
    assert finals, "30 秒内没有句子定稿，等价性断言是真空通过"
    assert any(x[0] == "asr_partial" for x in left), "没有草稿事件，链路没真正跑起来"
    assert sum(len(x[1]) for x in finals) > 0, "定稿文本全空"

    assert left == right, f"默认参数改变了事件序列：first diff at {_first_diff(left, right)}"
