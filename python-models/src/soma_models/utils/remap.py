from typing import Dict, Any, List


def remap(
    d: Dict[str, Any], rename_map: Dict[str, str], keys_to_delete: List[str]
) -> None:
    """Rename keys in `d` according to `rename_map` and delete `keys_to_delete`, in-place."""
    for key in keys_to_delete:
        d.pop(key, None)

    for old_key, new_key in rename_map.items():
        if old_key in d and old_key != new_key:
            d[new_key] = d.pop(old_key)
