import pytest
from soma_models.utils import flatten_dict, unflatten_dict


def test_flatten_simple():
    assert flatten_dict({"a": {"b": 1, "c": 2}}) == {"a.b": 1, "a.c": 2}


def test_flatten_deeply_nested():
    assert flatten_dict({"a": {"b": {"c": 3}}}) == {"a.b.c": 3}


def test_flatten_empty():
    assert flatten_dict({}) == {}


def test_flatten_already_flat():
    assert flatten_dict({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_flatten_preserves_empty_dict_value():
    assert flatten_dict({"a": {}}) == {"a": {}}


def test_flatten_mixed_depth():
    d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    assert flatten_dict(d) == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_unflatten_simple():
    assert unflatten_dict({"a.b": 1, "a.c": 2}) == {"a": {"b": 1, "c": 2}}


def test_unflatten_deeply_nested():
    assert unflatten_dict({"a.b.c": 3}) == {"a": {"b": {"c": 3}}}


def test_unflatten_empty():
    assert unflatten_dict({}) == {}


def test_unflatten_flat_keys():
    assert unflatten_dict({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_roundtrip():
    original = {"a": {"b": {"c": 1}}, "d": {"e": 2}, "f": 3}
    assert unflatten_dict(flatten_dict(original)) == original


def test_unflatten_conflict_value_then_dict():
    """A key can't be both a leaf value and a nested dict."""
    with pytest.raises(ValueError, match="Unflattening conflict"):
        unflatten_dict({"a.b": 1, "a.b.c": 2})


def test_unflatten_conflict_dict_then_value():
    """A key that's already a dict can't be overwritten with a scalar."""
    with pytest.raises(ValueError, match="Unflattening conflict"):
        unflatten_dict({"a.b.c": 2, "a.b": 1})


def test_custom_separator():
    assert flatten_dict({"a": {"b": 1}}, sep="/") == {"a/b": 1}
    assert unflatten_dict({"a/b": 1}, sep="/") == {"a": {"b": 1}}
