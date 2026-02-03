def flatten_dict(nested_dict, parent_key="", sep="."):
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and v:  # Check if v is a non-empty dictionary
            # Recursively flatten the nested dictionary
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # If v is not a dictionary or an empty dictionary, add it as is
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(flat_dict, sep="."):
    result_dict = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        d = result_dict
        for i, part in enumerate(
            parts[:-1]
        ):  # Iterate through all parts except the last one
            if part not in d:
                d[part] = {}
            elif not isinstance(d[part], dict):
                raise ValueError(
                    f"Unflattening conflict: key '{part}' in path '{key}' "
                    f"already exists with a non-dictionary value: {d[part]}"
                )
            d = d[part]

        last_part = parts[-1]
        if (
            last_part in d
            and isinstance(d[last_part], dict)
            and not isinstance(value, dict)
        ):
            raise ValueError(
                f"Unflattening conflict: key '{last_part}' in path '{key}' "
                f"is already a dictionary, but trying to assign a non-dictionary value: {value}"
            )

        d[last_part] = value
    return result_dict
