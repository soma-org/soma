from typing import Dict
import torch
import torch.nn as nn
from safetensors.torch import load


_LAYER_NORM_RENAMES = {
    "gamma": "weight",
    "beta": "bias",
}


def _remap_key(key: str) -> str:
    parts = key.rsplit(".", 1)
    if len(parts) == 2 and parts[1] in _LAYER_NORM_RENAMES:
        return f"{parts[0]}.{_LAYER_NORM_RENAMES[parts[1]]}"
    return key


def _find_linear_weight_keys(module: nn.Module, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for name, child in module.named_children():
        full = f"{prefix}{name}" if prefix else name
        if isinstance(child, nn.Linear):
            keys.add(f"{full}.weight")
        else:
            keys.update(_find_linear_weight_keys(child, f"{full}."))
    return keys


def load_safetensor_into(module: nn.Module, data: bytes) -> None:
    tensors = load(data)
    remapped: Dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        remapped[_remap_key(key)] = tensor

    linear_keys = _find_linear_weight_keys(module)
    for key in linear_keys:
        if key in remapped and remapped[key].ndim == 2:
            remapped[key] = remapped[key].t()

    module.load_state_dict(remapped)
