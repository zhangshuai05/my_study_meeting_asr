# -*- coding: utf-8 -*-
"""streaming_tuning 的算术自检：不加载模型。"""
import pytest

import streaming_tuning as st


class FakeSession:
    """五个属性故意先填成错的值。

    如果填成上游的正确值，"tune 没赋值" 与 "tune 赋了同样的值" 就无法区分，
    测试会在破坏时依然全绿。这里用错值，赋值与否一眼可判。
    """
    chunk_size_asr = [9, 9, 9]
    chunk_stride_samples = 111
    chunk_size_vad = 111
    encoder_chunk_look_back = 9
    decoder_chunk_look_back = 9


def test_default_is_identity():
    """不传参时必须把五个属性全部改写成上游默认值。"""
    session = st.tune(FakeSession())
    assert session.chunk_size_asr == [0, 10, 5]
    assert session.chunk_stride_samples == 9600
    assert session.chunk_size_vad == 600
    assert (session.encoder_chunk_look_back, session.decoder_chunk_look_back) == (4, 1)


@pytest.mark.parametrize("chunk,stride", [
    ((0, 5, 2), 4800),
    ((0, 8, 4), 7680),
    ((0, 10, 5), 9600),
    ((0, 16, 8), 15360),
])
def test_stride_derives_from_chunk(chunk, stride):
    """chunk_size 的单位是 60ms 帧，stride 必须同步推导，不能只改 chunk。"""
    session = st.tune(FakeSession(), chunk)
    assert session.chunk_size_asr == [0, chunk[1], chunk[2]]
    assert session.chunk_stride_samples == stride


def test_vad_chunk_follows_stride_by_default():
    """VAD 的 chunk_size 描述"喂进来一块有多长"，必须跟 stride 走。

    喂 300ms 却告诉 VAD 600ms，它的端点判定会整体晚一块。
    """
    session = st.tune(FakeSession(), (0, 5, 2))
    assert session.chunk_size_vad == 300


def test_vad_chunk_can_be_overridden():
    session = st.tune(FakeSession(), (0, 5, 2), vad_chunk_ms=600)
    assert session.chunk_size_vad == 600


def test_rejects_unverified_chunk():
    with pytest.raises(ValueError):
        st.tune(FakeSession(), (0, 7, 3))


def test_rejects_bad_look_back():
    with pytest.raises(ValueError):
        st.tune(FakeSession(), look_back=(0, 1))
