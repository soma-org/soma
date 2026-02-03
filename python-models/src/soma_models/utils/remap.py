from typing import Dict, Any, List


def remap(
    dict: Dict[str, Any], rename_map: Dict[str, str], keys_to_delete: List[str]
) -> None:
    pending_operations = []  # Stores tuples of (new_key_target, value_from_old_key)
    keys_to_delete_from_original = set(keys_to_delete)

    for old_key, new_key in rename_map.items():
        if old_key in dict:
            if old_key == new_key:
                continue
            value = dict[old_key]
            pending_operations.append((new_key, value))
            keys_to_delete_from_original.add(old_key)

    for key_to_del in keys_to_delete_from_original:
        dict.pop(key_to_del)

    for new_key, value in pending_operations:
        dict[new_key] = value
