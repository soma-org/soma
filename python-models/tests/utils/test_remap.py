from soma_models.utils import remap


def test_remap_basic_rename():
    d = {"old": 1, "keep": 2}
    remap(d, {"old": "new"}, [])
    assert d == {"new": 1, "keep": 2}


def test_remap_delete_keys():
    d = {"a": 1, "b": 2, "c": 3}
    remap(d, {}, ["a", "c"])
    assert d == {"b": 2}


def test_remap_rename_and_delete():
    d = {"old": 1, "trash": 2, "keep": 3}
    remap(d, {"old": "new"}, ["trash"])
    assert d == {"new": 1, "keep": 3}


def test_remap_noop_same_key():
    d = {"a": 1}
    remap(d, {"a": "a"}, [])
    assert d == {"a": 1}


def test_remap_missing_rename_key_ignored():
    d = {"a": 1}
    remap(d, {"nonexistent": "b"}, [])
    assert d == {"a": 1}


def test_remap_missing_delete_key_ignored():
    d = {"a": 1}
    remap(d, {}, ["nonexistent"])
    assert d == {"a": 1}


def test_remap_empty_everything():
    d = {}
    remap(d, {}, [])
    assert d == {}


def test_remap_multiple_renames():
    d = {"a": 1, "b": 2, "c": 3}
    remap(d, {"a": "x", "b": "y"}, [])
    assert d == {"x": 1, "y": 2, "c": 3}
