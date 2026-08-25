import pytest

from src.evaluation.tracking_metrics import (
    haversine_m,
    hota,
    id_switches,
    idf1,
    pixel_distance,
    reacquisition_topk,
    summarize_reacquisition,
)

BOX = [0.0, 0.0, 10.0, 10.0]


def test_haversine():
    assert haversine_m(0, 0, 0, 0) == pytest.approx(0.0)
    # one degree of latitude at the equator ~ 111.19 km
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111194.9, rel=1e-3)


def test_pixel_distance():
    assert pixel_distance([0, 0, 10, 10], [100, 0, 10, 10]) == pytest.approx(100.0)


def test_reacquisition_topk():
    assert reacquisition_topk([1, 3, None], 1) == pytest.approx(1 / 3)
    assert reacquisition_topk([1, 3, None], 5) == pytest.approx(2 / 3)
    assert reacquisition_topk([], 1) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        reacquisition_topk([1], 0)


def test_id_switches():
    aligned = [(0, "a"), (0, "a"), (0, "b"), (1, "x"), (0, "b")]
    assert id_switches(aligned) == 1  # gt 0: a -> b at frame 3
    assert id_switches([(0, "a"), (0, "a")]) == 0


def test_idf1_perfect():
    frames = [[(0, BOX)]] * 3
    assert idf1(frames, frames) == pytest.approx(1.0)


def test_idf1_no_predictions():
    frames = [[(0, BOX)]] * 3
    assert idf1(frames, [[]] * 3) == pytest.approx(0.0)


def test_idf1_one_swap():
    gt = [[(0, BOX)]] * 3
    pred = [[(0, BOX)], [(1, BOX)], [(0, BOX)]]
    # gt0 matches pred0 on 2 frames -> IDTP=2, detections 3/3 -> IDF1 = 2/3
    assert idf1(gt, pred) == pytest.approx(2 / 3)


def test_hota_perfect():
    frames = [[(0, BOX)]] * 5
    assert hota(frames, frames) == pytest.approx(1.0)


def test_hota_no_predictions():
    frames = [[(0, BOX)]] * 5
    assert hota(frames, [[]] * 5) == pytest.approx(0.0)


def test_hota_partial_recovery():
    # half the frames tracked perfectly, half missed entirely
    gt = [[(0, BOX)]] * 4
    pred = [[(0, BOX)], [(0, BOX)], [], []]
    h = hota(gt, pred, alphas=[0.5])
    assert 0.0 < h < 1.0


def test_summarize_reacquisition():
    results = [
        {"duration_s": 10, "rank_of_correct": 1, "n_candidates": 5, "drift_m": 12.0},
        {"duration_s": 10, "rank_of_correct": 4, "n_candidates": 5, "drift_m": None},
        {"duration_s": 60, "rank_of_correct": None, "n_candidates": 5, "drift_m": 90.0},
    ]
    s = summarize_reacquisition(results)
    assert s["per_duration"][0]["duration_s"] == 10
    assert s["per_duration"][0]["top1"] == pytest.approx(0.5)
    assert s["per_duration"][0]["top5"] == pytest.approx(1.0)
    assert s["per_duration"][0]["mean_drift_m"] == pytest.approx(12.0)
    assert s["per_duration"][1]["top1"] == pytest.approx(0.0)
    assert s["top1_by_duration"] == {"10s": 0.5, "60s": 0.0}
