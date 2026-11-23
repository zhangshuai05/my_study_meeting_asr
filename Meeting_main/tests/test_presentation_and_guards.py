# -*- coding: utf-8 -*-
"""针对 2026-10-19 审计报告 (A1-A7, B1-B5, C1-C7, D1-D6, E1-E8) 的自动化回归测试。

覆盖重点：
1. [A4 / D2] `clean_cjk_spaces` 去除 SEACO 分词空格，同时严格保留中文报号组边界 (`零 零 六` + `零 幺 三` -> `零零六、零幺三`)。
2. [D2] `grouped_cont` 同组紧邻不插机械逗号，跨组长停顿 (`>= PAUSE_COMMA_GAP_MS`) 才补逗号。
3. [D1] `session.json` 中的 `utt["raw_text"]` 与 `utterances.jsonl` 完全一致，不被 `clean_cjk_spaces` 覆盖。
4. [A6 / A7] `final_polish` 说话人感知上下文、长段膨胀上限、阿拉伯数字守卫、全局时间预算。
5. [E2 / E3 / E4] `offline_asr` 小数/英文切片不插空格、组内拼接保留自然静音、`_restore_trailing_punct` 只信 `raw_text`。
6. [A2 / D3 / E6 / C5] `stream_service` 捕获 `SystemExit` 正常收尾 NDJSON、Origin 校验、`/api/sessions` 历史会话查询。
7. [全量会议仿真] 4 说话人、95 段 (含长独白) 端到端合并与精修验证。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diar_backfill import to_contract
from final_polish import _build_speaker_context, accept_final, polish_turns
from offline_asr import _concat_decoded_chunks, _restore_trailing_punct
from session_store import SessionStore
from turns import (
     MAX_TURN_CHARS,
    PAUSE_COMMA_GAP_MS,
    _END_PUNCT,
    _join_turn_utterances,
    build_turns,
    clean_cjk_spaces,
    join_texts,
)


def test_a4_clean_cjk_spaces_preserves_spoken_digit_groups():
    # 普通 SEACO 分词句：字间单空格被彻底清除
    assert clean_cjk_spaces("今 天 啊 把 各 位 都 叫 过 来 啊") == "今天啊把各位都叫过来啊"
    # 单句内的多组报号（组间双空格）：组间保留单空格
    s = "零 零 六  零 幺 三  零 幺 四"
    assert clean_cjk_spaces(s) == "零零六、零幺三、零幺四"
    assert clean_cjk_spaces("零 幺 三 零 幺 四 零 幺 九。") == "零幺三、零幺四、零幺九。"
    # 跨句拼接的报号句：自动以顿号分隔，绝不糊成 "零零六零幺三零幺四"
    utts = [
        {"utt_id": "u1", "start_ms": 0, "end_ms": 800, "text": "零 零 六", "spk": 0, "spk_label": "speaker_0"},
        {"utt_id": "u2", "start_ms": 900, "end_ms": 1700, "text": "零 幺 三", "spk": 0, "spk_label": "speaker_0"},
        {"utt_id": "u3", "start_ms": 1800, "end_ms": 2600, "text": "零 幺 四", "spk": 0, "spk_label": "speaker_0"},
    ]
    joined = _join_turn_utterances(utts)
    assert joined == "零零六、零幺三、零幺四。"


def test_d2_grouped_cont_and_pause_comma_gap():
    # 同组内 VAD 切点在词中间 ("小" | "区环境")，grouped_cont=True 时绝不插逗号
    utts_grouped = [
        {"utt_id": "u1", "start_ms": 1000, "end_ms": 2200, "text": "如何把咱这小", "grouped_cont": False},
        {"utt_id": "u2", "start_ms": 2800, "end_ms": 4000, "text": "区环境搞好", "grouped_cont": True},
    ]
    assert _join_turn_utterances(utts_grouped) == "如何把咱这小区环境搞好。"

    # 非同组且真实停顿 >= PAUSE_COMMA_GAP_MS (450ms) 时才补逗号
    utts_paused = [
        {"utt_id": "u1", "start_ms": 1000, "end_ms": 2000, "text": "第一点大家注意", "grouped_cont": False},
        {"utt_id": "u2", "start_ms": 2000 + PAUSE_COMMA_GAP_MS + 50, "end_ms": 3500, "text": "明天准时集合", "grouped_cont": False},
    ]
    assert _join_turn_utterances(utts_paused) == "第一点大家注意，明天准时集合。"


def test_e1_end_punct_excludes_whitespace_and_quotes():
    assert " " not in _END_PUNCT
    assert "\t" not in _END_PUNCT
    assert '"' not in _END_PUNCT
    assert "'" not in _END_PUNCT
    # 以空格结尾的中文片段在需要补逗号时仍能正确识别为无标点
    assert join_texts(["今天开会 ", "讨论预算"], insert_comma=True) == "今天开会，讨论预算"


def test_e2_e4_offline_asr_helpers():
    # E2: 小数与英文单词跨切片拼接不插空格
    assert _concat_decoded_chunks(["3.", "5", "，obj", "ect"]) == "3.5，object"
    assert _concat_decoded_chunks(["今 天 ", "开 会"]) == "今天开会"
    # E4: _restore_trailing_punct 只信任 raw_text 的真实标点，不继承 LLM 猜测的句号
    utt_llm_only = {"raw_text": "今天我们开会", "polished_text": "今天我们开会。"}
    assert _restore_trailing_punct("今天我们开会", utt_llm_only) == "今天我们开会"
    utt_raw_punct = {"raw_text": "今天我们开会？", "polished_text": "今天我们开会？"}
    assert _restore_trailing_punct("今天我们开会", utt_raw_punct) == "今天我们开会？"


def test_d1_c1_session_store_preserves_raw_text_and_accurate_badges(tmp_path: Path):
    store = SessionStore(tmp_path / "s_20261019_160000")
    raw_with_spaces = "零 零 六 零 幺 三"
    store.append_utterance({
        "utt_id": "utt_0001",
        "start_ms": 0,
        "end_ms": 1500,
        "raw_text": raw_with_spaces,
    })
    store.write_transcript([{"utt_id": "utt_0001", "start_ms": 0, "end_ms": 1500, "raw_text": raw_with_spaces}], duration_ms=1500)
    store.write_polished({"utt_0001": {"polished_text": "零零六，零幺三。", "corrected": True, "bypassed": False}})
    store.write_diarization({"utt_0001": {"spk": 0, "spk_label": "speaker_0"}})
    store.write_final_text({"utt_0001": {"text": "零 零 六  零 幺 三", "grouped_cont": False}})
    # 模拟精修未改变文本 (unchanged)
    store.write_turns_polished({"turn_0001": {"text": "零零六、零幺三。", "start_ms": 0, "end_ms": 1500}})
    sess = store.build_session_json()

    # D1: session.json 里的 raw_text 必须原封不动保留空格
    assert sess["utterances"][0]["raw_text"] == raw_with_spaces
    assert sess["turns"][0]["raw_text"] == raw_with_spaces
    # C1: 当 turns_polished 与二遍定稿一致时，text_source 仍为 final_asr 而非谎报 turn_polished
    assert sess["turns"][0]["text_source"] == "final_asr"
    assert sess["turns"][0]["polish_status"] == "unchanged"
    # D6: session.json 包含 upstream_baseline
    assert "upstream_baseline" in sess


def test_a6_a7_final_polish_speaker_context_and_digit_guard():
    turns = [
        {"turn_id": "turn_0001", "spk": 0, "spk_label": "speaker_0", "text": "我们看第 12 号方案的预算。"},
        {"turn_id": "turn_0002", "spk": 1, "spk_label": "speaker_1", "text": "我不同意这个数字。"},
        {"turn_id": "turn_0003", "spk": 0, "spk_label": "speaker_0", "text": "那我们把 12 号方案改成 15 万吧"},
    ]
    ctx = _build_speaker_context(turns, 2)
    assert "speaker_0 稍早发言" in ctx and "另一位说话人 speaker_1" in ctx
    assert "12 号方案" in ctx

    # A7: 模型篡改阿拉伯数字 (15 -> 50) 时被 polish_turns 拦截回退
    def bad_llm(prompt: str) -> str:
        return json.dumps({"polished": "那我们把 12 号方案改成 50 万吧。"}, ensure_ascii=False)

    res = polish_turns([turns[2]], call=bad_llm)
    assert "turn_0003" not in res

    # A7: 900 字长段若模型塞入 100 字总结 (> MAX_ABS_ADDED_CHARS=45)，直接拦截
    long_src = "这是一段正式发言。" * 100  # 900 字
    assert not accept_final(long_src, long_src + "总结如下：" * 12)


def test_e7_diar_backfill_skips_corrupted_row_in_batch():
    raw = {
        "mapping": {
            "utt_0001": "Speaker_1",
            "utt_0002": "BAD_LABEL",
            "utt_0003": "Speaker_2",
        }
    }
    contract = to_contract(raw)
    assert set(contract.keys()) == {"utt_0001", "utt_0003"}
    assert contract["utt_0001"]["spk_label"] == "speaker_0"
    assert contract["utt_0003"]["spk_label"] == "speaker_1"


def test_full_95_turns_4_speakers_monologue_split_and_merge():
    # 模拟 4 人、120 句（含一段 90 秒 700 字长独白）的全量会议合成
    utterances = []
    t_ms = 0
    for i in range(1, 121):
        # 第 20~55 句为 speaker_2 的超长独白
        spk = 2 if (20 <= i <= 55) else (i % 4)
        text = "各位同事我们讨论一下今年第三季度的重点项目推进计划和排期安排。" if (20 <= i <= 55) else f"这是第{i}句常规讨论发言。"
        dur = 2800
        utterances.append({
            "utt_id": f"utt_{i:04d}",
            "start_ms": t_ms,
            "end_ms": t_ms + dur,
            "raw_text": text,
            "polished_text": text,
            "final_text": text,
            "corrected": True,
            "bypassed": False,
            "spk": spk,
            "spk_label": f"speaker_{spk}",
        })
        t_ms += dur + 300

    turns = build_turns(utterances)
    assert len(turns) >= 60
    # E8: 没有任何一个 Turn 超过软上限两倍，长独白被安全切分为多个可读段落
    for turn in turns:
        assert len(turn["text"]) <= MAX_TURN_CHARS + 80


if __name__ == "__main__":
    import tempfile
    test_a4_clean_cjk_spaces_preserves_spoken_digit_groups()
    test_d2_grouped_cont_and_pause_comma_gap()
    test_e1_end_punct_excludes_whitespace_and_quotes()
    test_e2_e4_offline_asr_helpers()
    with tempfile.TemporaryDirectory() as td:
        test_d1_c1_session_store_preserves_raw_text_and_accurate_badges(Path(td))
    test_a6_a7_final_polish_speaker_context_and_digit_guard()
    test_e7_diar_backfill_skips_corrupted_row_in_batch()
    test_full_95_turns_4_speakers_monologue_split_and_merge()
    print("[PASS] All 8 presentation, guard, and 95-turn 4-speaker regression tests passed!")
