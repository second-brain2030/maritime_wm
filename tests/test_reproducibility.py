import random

from src.utils.reproducibility import content_addressed_key, seed_everything


def test_key_stable_and_order_insensitive():
    a = content_addressed_key({"a": 1, "b": [1, 2]})
    b = content_addressed_key({"b": [1, 2], "a": 1})
    assert a == b
    assert len(a) == 64


def test_key_sensitive_to_values():
    assert content_addressed_key({"a": 1}) != content_addressed_key({"a": 2})


def test_key_handles_non_json():
    # datetimes etc. are stringified deterministically
    class Foo:
        def __str__(self):
            return "foo"

    k = content_addressed_key({"x": Foo()})
    assert len(k) == 64


def test_seed_everything_deterministic():
    seed_everything(7)
    x = random.random()
    seed_everything(7)
    assert random.random() == x
