import pytest

from utils.registry import Registry


def test_register_create():
    reg = Registry("test")
    reg.register("a", lambda: 1)
    assert reg.create("a") == 1
    assert reg.names() == ["a"]


def test_decorator_style():
    reg = Registry("test")

    @reg.register("f")
    def f():
        return 2

    assert reg.create("f") == 2


def test_unknown_raises():
    reg = Registry("test")
    with pytest.raises(KeyError):
        reg.create("missing")


def test_contains():
    reg = Registry("test")
    reg.register("a", lambda: 0)
    assert "a" in reg
    assert "b" not in reg


def test_create_with_kwargs():
    reg = Registry("test")
    reg.register("add", lambda x, y=0: x + y)
    assert reg.create("add", x=1, y=2) == 3
