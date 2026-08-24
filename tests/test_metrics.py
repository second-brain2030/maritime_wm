import pytest

from evaluation.reid_metrics import cmc, mean_average_precision, rank_gallery


def test_cmc():
    scores = {
        "q1": {"g1": 0.9, "g2": 0.8, "g3": 0.7},
        "q2": {"g1": 0.5, "g2": 0.6, "g3": 0.9},
    }
    gt = {"q1": {"g2"}, "q2": {"g1"}}
    c = cmc(scores, gt, max_rank=3)
    assert c[0] == pytest.approx(0.0)  # no query has the relevant item at rank 1
    assert c[1] == pytest.approx(0.5)  # q1 yes, q2 no
    assert c[2] == pytest.approx(1.0)


def test_map_single_query():
    scores = {"q1": {"g1": 0.9, "g2": 0.8, "g3": 0.7}}
    gt = {"q1": {"g2", "g3"}}
    # ranks: g1 (miss), g2 (hit, prec 1/2), g3 (hit, prec 2/3) -> AP = 7/12
    assert mean_average_precision(scores, gt) == pytest.approx(7 / 12)


def test_map_empty_returns_zero():
    assert mean_average_precision({}, {}) == 0.0


def test_rank_gallery_descending():
    assert rank_gallery({"a": 0.1, "b": 0.9, "c": 0.5}) == ["b", "c", "a"]
