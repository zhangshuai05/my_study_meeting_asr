# -*- coding: utf-8 -*-
"""流式参数覆盖：只改实例属性，不改上游一个字节。

理由：`StreamingSession` 的 chunk_size_asr / chunk_stride_samples / chunk_size_vad /
encoder_chunk_look_back / decoder_chunk_look_back 全是实例属性，构造后覆写即可生效。
"""
from __future__ import annotations

CHUNK_CHOICES = {
    (0, 5, 2): "300ms",
    (0, 8, 4): "480ms",
    (0, 10, 5): "600ms",
    (0, 16, 8): "960ms",
}
DEFAULT_CHUNK = (0, 10, 5)  # 与上游硬编码值一致，改参数前先保证这里相等
SAMPLE_PER_FRAME = 960      # chunk_size 的单位是 60ms 帧 = 960 样本 @16kHz


def tune(session, chunk=DEFAULT_CHUNK, look_back=(4, 1), vad_chunk_ms=None):
    """把已构造的 StreamingSession 调到指定 chunk 档位，返回同一个对象。

    vad_chunk_ms 必须跟 stride 对齐：fsmn-vad 的 chunk_size 描述"喂进来的一块有多长"，
    喂 300ms 却告诉它 600ms 会让它的端点判定整体晚一块。默认跟随 chunk。
    """
    chunk = tuple(chunk)
    if chunk not in CHUNK_CHOICES:
        raise ValueError(
            f"不支持的 chunk {chunk}，只允许 {sorted(CHUNK_CHOICES)}"
        )
    if look_back[0] < 1 or look_back[1] < 1:
        raise ValueError(f"look_back 必须为正整数对，收到 {look_back}")
    session.chunk_size_asr = [0, chunk[1], chunk[2]]
    session.chunk_stride_samples = chunk[1] * SAMPLE_PER_FRAME
    session.chunk_size_vad = chunk[1] * 60 if vad_chunk_ms is None else vad_chunk_ms
    session.encoder_chunk_look_back, session.decoder_chunk_look_back = look_back
    return session


def make_factory(chunk=DEFAULT_CHUNK, look_back=(4, 1), vad_chunk_ms=None,
                 max_end_silence_time=None):
    """给 run_offline.run(session_factory=...) 用的工厂：先构造再调参数。

    max_end_silence_time 传 None 时不覆盖，交给 StreamingSession 自己的默认值。
    不在这里重复写 1000 —— 上游改默认值时，这里会悄悄分叉。
    """
    def factory(session_wav, models):
        from streaming_asr_engine import StreamingSession
        extra = {}
        if max_end_silence_time is not None:
            extra["max_end_silence_time"] = max_end_silence_time
        session = StreamingSession(
            str(session_wav),
            asr_model=models["stream_asr"],
            vad_model=models["vad"],
            **extra,
        )
        return tune(session, chunk, look_back, vad_chunk_ms)
    return factory
