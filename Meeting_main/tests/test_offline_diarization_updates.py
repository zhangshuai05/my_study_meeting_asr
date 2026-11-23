# -*- coding: utf-8 -*-
import sys
from types import ModuleType

import numpy as np
import pytest

import bootstrap

bootstrap.install()
_paths_after_bootstrap = list(sys.path)

import offline_diarizer as diarizer  # noqa: E402
import run_3dspeaker_pipeline as pipeline  # noqa: E402

sys.path[:] = _paths_after_bootstrap


def test_cluster_merge_joins_split_identity_and_compacts_labels():
    embeddings = np.asarray([
        [1.00, 0.00], [0.99, 0.10], [0.98, 0.20],
        [0.00, 1.00], [0.10, 0.99], [0.20, 0.98],
        [0.99, 0.05], [0.98, 0.12],
    ], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1, 4, 6], dtype=int)

    merged, count = pipeline.merge_similar_clusters(
        embeddings, labels, merge_cos_th=0.58, min_cluster_size=3
    )

    assert count == 2
    assert set(merged.tolist()) == {0, 1}
    assert merged[0] == merged[6] == merged[7]


def test_spectral_cluster_rejects_nonpositive_exact_speaker_count():
    with pytest.raises(ValueError, match="正整数"):
        pipeline.spectral_cluster(np.eye(4, dtype=np.float32), 0)


def test_spectral_cluster_keeps_empty_input_as_empty_labels():
    labels, count, _ = pipeline.spectral_cluster(np.empty((0, 2), dtype=np.float32))

    assert labels.size == 0
    assert count == 1


def test_spectral_cluster_rejects_exact_speaker_count_without_embeddings():
    with pytest.raises(ValueError, match="有效嵌入数"):
        pipeline.spectral_cluster(np.empty((0, 2), dtype=np.float32), 2)


def test_min_speakers_is_honored_with_two_embeddings():
    labels, count, _ = pipeline.spectral_cluster(
        np.eye(2, dtype=np.float32), min_speakers=2
    )

    assert count == 2
    assert set(labels.tolist()) == {0, 1}


def test_min_speakers_above_available_embeddings_raises():
    with pytest.raises(ValueError, match="有效嵌入数"):
        pipeline.spectral_cluster(np.eye(2, dtype=np.float32), min_speakers=3)


def test_min_speakers_with_empty_embeddings_raises():
    with pytest.raises(ValueError, match="有效嵌入数"):
        pipeline.spectral_cluster(np.empty((0, 2), dtype=np.float32), min_speakers=1)


@pytest.mark.parametrize("empty_at", ["vad", "embedding"])
def test_backfill_enforces_speaker_bounds_without_embeddings(monkeypatch, empty_at):
    monkeypatch.setattr(diarizer.sf, "read", lambda *args, **kwargs: (
        np.zeros(16000, dtype=np.float32), 16000
    ))
    monkeypatch.setattr(diarizer, "_normalize_audio", lambda audio, rate, target: (audio, None))

    v11 = ModuleType("claude_a2_v11")
    v11._vad_segs = lambda model, wav: [] if empty_at == "vad" else [(0.0, 1.0)]
    v12 = ModuleType("claude_a2_v12")
    monkeypatch.setitem(sys.modules, "claude_a2_v11", v11)
    monkeypatch.setitem(sys.modules, "claude_a2_v12", v12)
    if empty_at == "embedding":
        monkeypatch.setattr(pipeline, "chunk_windows", lambda vad: [(0.0, 1.0)])
        monkeypatch.setattr(
            pipeline, "embed_chunks",
            lambda model, audio, rate, wins: (np.empty((0, 2), dtype=np.float32), []),
        )

    with pytest.raises(ValueError, match="有效嵌入数"):
        diarizer.backfill_speakers(
            "unused.wav",
            [{"utt_id": "u1", "start_ms": 0, "end_ms": 1000}],
            models={"vad": object(), "spk": object()},
            mode="vad",
            min_speakers=1,
        )


def test_max_speakers_can_raise_the_default_cap(monkeypatch):
    import scipy.cluster.vq
    import scipy.linalg

    observed = {}
    monkeypatch.setattr(
        scipy.linalg, "eigh",
        lambda matrix: (np.arange(len(matrix), dtype=float), np.eye(len(matrix))),
    )
    monkeypatch.setattr(
        scipy.cluster.vq, "kmeans2",
        lambda vectors, count, **kwargs: (np.zeros((count, vectors.shape[1])),
                                          np.arange(len(vectors)) % count),
    )

    def choose_count(values, min_k, max_k):
        observed["max_k"] = max_k
        return max_k

    monkeypatch.setattr(pipeline, "eigengap_k", choose_count)
    pipeline.spectral_cluster(np.ones((12, 2), dtype=np.float32), max_speakers=10,
                              merge_cos_th=0.0)

    assert observed["max_k"] == 10


def test_spectral_cluster_accepts_one_cluster_per_available_embedding(monkeypatch):
    import scipy.cluster.vq
    import scipy.linalg

    monkeypatch.setattr(
        scipy.linalg, "eigh",
        lambda matrix: (np.arange(len(matrix), dtype=float), np.eye(len(matrix))),
    )
    monkeypatch.setattr(
        scipy.cluster.vq, "kmeans2",
        lambda vectors, count, **kwargs: (np.zeros((count, vectors.shape[1])),
                                          np.arange(len(vectors))),
    )

    labels, count, _ = pipeline.spectral_cluster(
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
        n_clusters=3,
    )

    assert count == 3
    assert len(set(labels.tolist())) == 3


def test_microclusters_are_absorbed_as_whole_clusters():
    embeddings = np.asarray([
        [1.00, 0.00], [0.99, 0.10], [0.98, 0.20],
        [0.00, 1.00], [0.10, 0.99], [0.20, 0.98],
        [0.95, 0.31], [0.96, 0.28],
    ], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1, 4, 4], dtype=int)

    merged, count = pipeline.merge_similar_clusters(
        embeddings, labels, merge_cos_th=0.99, min_cluster_size=3
    )

    assert count == 2
    assert merged[6] == merged[7] == merged[0]


def test_microclusters_merge_when_no_cluster_meets_the_size_threshold():
    embeddings = np.eye(4, dtype=np.float32)
    labels = np.asarray([0, 1, 2, 3], dtype=int)

    merged, count = pipeline.merge_similar_clusters(
        embeddings, labels, merge_cos_th=0.99, min_cluster_size=3, min_k=3
    )

    assert count == 3
    assert max(np.bincount(merged)) == 2


def test_turn_vote_uses_long_windows_and_short_windows_only_as_fallback(monkeypatch):
    long_windows = [(0.0, 1.5), (1.5, 3.0)]
    short_windows = [(0.0, 0.6), (0.15, 0.75)]
    long_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    short_embeddings = long_embeddings.copy()
    observed = {}

    def vote(windows, labels, turns, fallback=None):
        observed["windows"] = windows
        observed["fallback"] = fallback
        return [0], None

    monkeypatch.setattr(diarizer.sf, "read", lambda *args, **kwargs: (
        np.zeros(4 * 16000, dtype=np.float32), 16000
    ))
    monkeypatch.setattr(diarizer, "_normalize_audio", lambda audio, rate, target: (audio, None))
    monkeypatch.setattr(pipeline, "DENSE_RESCUE_ENABLED", False, raising=False)
    monkeypatch.setattr(pipeline, "chunk_windows", lambda vad: long_windows, raising=False)
    monkeypatch.setattr(
        pipeline, "embed_chunks", lambda model, audio, rate, wins: (long_embeddings, long_windows),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline, "spectral_cluster",
        lambda embeddings, n_clusters=None, **kwargs: (
            observed.update({"n_clusters": n_clusters}) or np.asarray([0, 1]),
            2,
            np.asarray([]),
        ),
        raising=False,
    )
    monkeypatch.setattr(pipeline, "turns_to_speakers", vote, raising=False)

    v11 = ModuleType("claude_a2_v11")
    v11._vad_segs = lambda model, wav: [(0.0, 4.0)]
    v11._centroids = lambda embeddings, labels: np.asarray([[1.0, 0.0], [0.0, 1.0]])
    v12 = ModuleType("claude_a2_v12")
    v12._refine_windows = lambda audio, rate, vad, model: (short_embeddings, [0.0, 0.15])
    v12._refine_labels = lambda embeddings, anchors, labels, times: np.asarray([0, 1])
    v12.REFINE_WIN_S = 0.6
    v12.REFINE_HOP_S = 0.15
    monkeypatch.setitem(sys.modules, "claude_a2_v11", v11)
    monkeypatch.setitem(sys.modules, "claude_a2_v12", v12)
    monkeypatch.setenv("DIAR_NUM_SPEAKERS", "2")

    diarizer.backfill_speakers(
        "unused.wav",
        [{"utt_id": "u1", "start_ms": 0, "end_ms": 3000}],
        models={"vad": object(), "spk": object()},
        mode="vad",
    )

    assert observed["windows"] == long_windows
    assert observed["fallback"][0] == short_windows
    assert observed["fallback"][1].tolist() == [0, 1]
    assert observed["n_clusters"] == 2


def test_explicit_speaker_count_disables_dense_identity_rescue(monkeypatch):
    long_windows = [(0.0, 1.5), (1.5, 3.0)]
    short_embeddings = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    long_embeddings = short_embeddings.copy()
    observed = {"dense_called": False}

    def vote(windows, labels, turns, fallback=None):
        observed["long_labels"] = labels.tolist()
        return [max(labels.tolist())], None

    monkeypatch.setattr(diarizer.sf, "read", lambda *args, **kwargs: (
        np.zeros(4 * 16000, dtype=np.float32), 16000
    ))
    monkeypatch.setattr(diarizer, "_normalize_audio", lambda audio, rate, target: (audio, None))
    monkeypatch.setattr(pipeline, "DENSE_RESCUE_ENABLED", True, raising=False)
    monkeypatch.setattr(pipeline, "chunk_windows", lambda vad: long_windows, raising=False)
    monkeypatch.setattr(
        pipeline, "embed_chunks", lambda model, audio, rate, wins: (long_embeddings, long_windows),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline, "spectral_cluster",
        lambda embeddings, n_clusters=None, **kwargs: (
            np.asarray([0, 1]), n_clusters, np.asarray([])
        ),
        raising=False,
    )
    monkeypatch.setattr(pipeline, "turns_to_speakers", vote, raising=False)

    def dense_rescue(embeddings):
        observed["dense_called"] = True
        return {
            "anchors": np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            "anchor_labels": [0, 1, 2],
            "k": 3,
            "silhouette": 0.4,
            "n_sampled": 12,
        }

    monkeypatch.setattr(pipeline, "dense_identity_estimate", dense_rescue, raising=False)
    v11 = ModuleType("claude_a2_v11")
    v11._vad_segs = lambda model, wav: [(0.0, 4.0)]
    v11._centroids = lambda embeddings, labels: np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    v12 = ModuleType("claude_a2_v12")
    v12._refine_windows = lambda audio, rate, vad, model: (short_embeddings, [0.0, 0.15])
    v12._refine_labels = lambda embeddings, anchors, labels, times: np.asarray(
        [0, 2] if len(anchors) == 3 else [0, 1]
    )
    v12.REFINE_WIN_S = 0.6
    v12.REFINE_HOP_S = 0.15
    monkeypatch.setitem(sys.modules, "claude_a2_v11", v11)
    monkeypatch.setitem(sys.modules, "claude_a2_v12", v12)

    mapping = diarizer.backfill_speakers(
        "unused.wav",
        [{"utt_id": "u1", "start_ms": 0, "end_ms": 3000}],
        models={"vad": object(), "spk": object()},
        mode="vad",
        num_speakers=2,
    )

    assert observed["dense_called"] is False
    assert observed["long_labels"] == [0, 1]
    assert mapping == {"u1": "Speaker_2"}


def test_short_turn_is_not_split_into_subturns():
    turn = {"utt_id": "u1", "start": 0.0, "end": 4.4}
    windows = [(i * 0.15, i * 0.15 + 0.6) for i in range(27)]
    labels = np.asarray([0] * 8 + [1] * 19, dtype=int)

    segments, sub_turns = diarizer._refined_turn_segments(
        [turn], windows, labels, [0], 0.15, 2.5, 0.5,
        rE=np.asarray([[1.0, 0.0]] * 8 + [[0.0, 1.0]] * 19),
        anchors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        anchor_labels=[0, 1],
    )

    assert sub_turns == {}
    assert len(segments["u1"]) == 1
    assert segments["u1"][0]["is_main"] is True


def test_short_low_confidence_island_is_smoothed():
    turns = [
        {"utt_id": "a", "start": 0.0, "end": 10.0},
        {"utt_id": "b", "start": 10.1, "end": 11.2},
        {"utt_id": "c", "start": 11.3, "end": 20.0},
    ]
    scores = np.asarray([[0.8, 0.2], [0.69, 0.65], [0.2, 0.8]], dtype=np.float32)

    assert diarizer._smooth_turn_speakers(turns, [1, 0, 1], scores, [0, 1]) == [1, 1, 1]


def test_turn_without_long_window_coverage_keeps_fallback_identity():
    turns = [
        {"utt_id": "a", "start": 0.0, "end": 10.0},
        {"utt_id": "b", "start": 10.1, "end": 11.0},
        {"utt_id": "c", "start": 11.1, "end": 20.0},
    ]
    scores = diarizer._turn_embedding_scores(
        turns,
        [(0.0, 10.0), (11.1, 20.0)],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [0, 1],
    )

    assert np.array_equal(scores[1], np.zeros(2, dtype=np.float32))
    assert diarizer._smooth_turn_speakers(turns, [1, 0, 1], scores, [0, 1]) == [1, 0, 1]


def test_confident_subturn_is_preserved_but_requires_long_parent_turn():
    turn = {"utt_id": "u1", "start": 0.0, "end": 6.6}
    windows = [(i * 0.15, i * 0.15 + 0.6) for i in range(41)]
    labels = np.asarray([0] * 20 + [1] * 21, dtype=int)
    embeddings = np.asarray([[1.0, 0.0]] * 20 + [[0.0, 1.0]] * 21)

    _, sub_turns = diarizer._refined_turn_segments(
        [turn], windows, labels, [0], 0.15, 2.5, 0.5,
        rE=embeddings,
        anchors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        anchor_labels=[0, 1],
    )

    assert sub_turns["u1"]
    assert sub_turns["u1"][0]["end_ms"] - sub_turns["u1"][0]["start_ms"] >= 2500


def test_subturn_without_cosine_margin_is_not_returned():
    turn = {"utt_id": "u1", "start": 0.0, "end": 6.6}
    windows = [(i * 0.15, i * 0.15 + 0.6) for i in range(41)]
    labels = np.asarray([0] * 20 + [1] * 21, dtype=int)
    embeddings = np.asarray([[1.0, 0.0]] * 20 + [[0.75, 0.66]] * 21)

    _, sub_turns = diarizer._refined_turn_segments(
        [turn], windows, labels, [0], 0.15, 2.5, 0.5,
        rE=embeddings,
        anchors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        anchor_labels=[0, 1],
    )

    assert sub_turns == {}


def _unit_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def test_fisher_merge_preserves_distinct_speakers_under_shared_room_bias():
    """Distinct speakers sharing a strong room/channel cone (raw cos ~0.65) must NOT merge,
    while two sub-clusters of the same speaker (near vs far) MUST merge."""
    rng = np.random.RandomState(7)
    dim = 64
    room_bias = np.zeros(dim, dtype=np.float32)
    room_bias[0] = 1.15  # strong shared channel vector

    s0 = np.zeros(dim, dtype=np.float32); s0[1] = 0.85
    s1 = np.zeros(dim, dtype=np.float32); s1[2] = 0.85
    s2 = np.zeros(dim, dtype=np.float32); s2[3] = 0.85
    s2_shift = s2.copy(); s2_shift[4] = 0.15  # mild distance shift of same speaker s2

    c0 = _unit_rows(room_bias + s0 + 0.06 * rng.randn(30, dim).astype(np.float32))
    c1 = _unit_rows(room_bias + s1 + 0.06 * rng.randn(30, dim).astype(np.float32))
    c2a = _unit_rows(room_bias + s2 + 0.06 * rng.randn(20, dim).astype(np.float32))
    c2b = _unit_rows(room_bias + s2_shift + 0.06 * rng.randn(20, dim).astype(np.float32))
    E = np.vstack([c0, c1, c2a, c2b])
    labels = np.asarray([0] * 30 + [1] * 30 + [2] * 20 + [3] * 20, dtype=int)

    # Confirm raw centroid cosine between distinct speakers c0 and c1 is > 0.60 (would fail old 0.58!)
    m0 = c0.mean(axis=0); m0 /= np.linalg.norm(m0)
    m1 = c1.mean(axis=0); m1 /= np.linalg.norm(m1)
    assert float(m0 @ m1) > 0.60

    merged, count = pipeline.merge_similar_clusters(E, labels)

    assert count == 3
    assert merged[0] != merged[30]
    assert merged[0] != merged[60]
    assert merged[30] != merged[60]
    assert merged[60] == merged[80]


def test_spectral_cluster_recovers_single_speaker_from_isotropic_cloud():
    rng = np.random.RandomState(11)
    dim = 64
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    E = _unit_rows(base + 0.08 * rng.randn(60, dim).astype(np.float32))

    labels, count, _ = pipeline.spectral_cluster(E)

    assert count == 1
    assert set(labels.tolist()) == {0}


def test_reconstruct_global_activity_aligns_permuted_overlapping_windows():
    """Two overlapping 10s windows with SWAPPED local speaker indices must align cleanly."""
    fr = pipeline.SEG_FRAMES
    half = fr // 2
    # Window 0 [0..10s]: first half local spk 0 (Alice), second half local spk 1 (Bob)
    p0 = np.zeros((fr, 7), dtype=np.float32)
    p0[:half, 1] = 1.0  # class 1 = local spk 0
    p0[half:, 2] = 1.0  # class 2 = local spk 1
    # Window 1 [5..15s]: SWAPPED indices! first half local spk 1 (Bob -> wait: class 1 = Bob!), second half local spk 0 (Alice -> class 2)
    p1 = np.zeros((fr, 7), dtype=np.float32)
    p1[:half, 1] = 1.0  # local spk 0 is NOW Bob!
    p1[half:, 2] = 1.0  # local spk 1 is NOW Alice!

    alice = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    bob = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    bal_mean = 0.5 * (alice + bob)
    bal_cents = _unit_rows(np.stack([alice - bal_mean, bob - bal_mean]))

    win_embs = np.zeros((2, 3, 3), dtype=np.float32)
    win_embs[0, 0] = alice
    win_embs[0, 1] = bob
    win_embs[1, 0] = bob    # swapped in window 1!
    win_embs[1, 1] = alice  # swapped in window 1!

    pw = {
        "win_starts_s": np.asarray([0.0, 5.0], dtype=np.float32),
        "win_valid_frames": np.asarray([fr, fr], dtype=np.int32),
        "win_probs": np.stack([p0, p1], axis=0),
        "win_embs": win_embs,
        "win_has_emb": np.asarray([[True, True, False], [True, True, False]], dtype=bool),
        "win_clean_s": np.asarray([[5.0, 5.0, 0.0], [5.0, 5.0, 0.0]], dtype=np.float32),
        "win_active_s": np.asarray([[5.0, 5.0, 0.0], [5.0, 5.0, 0.0]], dtype=np.float32),
        "frame_s": pipeline.SEG_FRAME_S,
    }

    act, _, _, fs = pipeline.reconstruct_global_activity(pw, 2, bal_cents, bal_mean)
    # 2.5s -> Alice (global 0); 7.5s (overlap region!) -> Bob (global 1); 12.5s -> Alice (global 0)
    assert int(np.argmax(act[int(2.5 / fs)])) == 0
    assert int(np.argmax(act[int(7.5 / fs)])) == 1
    assert float(act[int(7.5 / fs), 1]) > 0.95
    assert int(np.argmax(act[int(12.5 / fs)])) == 0


def test_activity_turn_segments_extracts_confident_subturn_and_rejects_weak_blip():
    fs = 0.02  # 50 Hz -> 500 frames = 10s
    act = np.zeros((500, 2), dtype=np.float32)
    # 0..3.5s spk 0 strong; 3.5..7.0s spk 1 strong (3.5s >= 2.5s); 7.0..10.0s spk 0 strong
    act[:175, 0] = 0.95
    act[175:350, 1] = 0.92
    act[350:, 0] = 0.95

    turn = {"utt_id": "u1", "start": 0.0, "end": 10.0}
    _, sub_turns = diarizer._activity_turn_segments(
        [turn], act, [0], fs, min_subturn_s=2.5, min_segment_s=0.5,
    )
    assert len(sub_turns["u1"]) == 1
    assert sub_turns["u1"][0]["speaker"] == "Speaker_2"
    assert sub_turns["u1"][0]["end_ms"] - sub_turns["u1"][0]["start_ms"] >= 3000

    # Weak margin (0.52 vs 0.45 = 0.07 < 0.15) must NOT trigger sub_turn
    weak_act = np.zeros((500, 2), dtype=np.float32)
    weak_act[:175, 0] = 0.95
    weak_act[175:350, 0] = 0.45
    weak_act[175:350, 1] = 0.52
    weak_act[350:, 0] = 0.95
    _, weak_subs = diarizer._activity_turn_segments(
        [turn], weak_act, [0], fs, min_subturn_s=2.5, min_segment_s=0.5,
    )
    assert weak_subs == {}


def test_short_confident_interjection_is_not_overwritten_by_smoothing():
    turns = [
        {"utt_id": "a", "start": 0.0, "end": 8.0},
        {"utt_id": "b", "start": 8.2, "end": 9.1},   # 0.9s confident interjection!
        {"utt_id": "c", "start": 9.3, "end": 16.0},
    ]
    scores = np.asarray([[0.90, 0.05], [0.08, 0.88], [0.92, 0.04]], dtype=np.float32)

    assert diarizer._smooth_turn_speakers(turns, [0, 1, 0], scores, [0, 1]) == [0, 1, 0]
