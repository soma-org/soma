import os
from typing import Dict, Union, List, TypeVar, Generic, Tuple
from safetensors.torch import save_file, load_file, save, load
from soma_models.utils import remap

import torch
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)

# Canonical safetensor key ↔ torch state_dict key
_LAYER_NORM_TO_TORCH: Dict[str, str] = {"gamma": "weight", "beta": "bias"}
_LAYER_NORM_TO_SAFETENSOR: Dict[str, str] = {v: k for k, v in _LAYER_NORM_TO_TORCH.items()}


class Serde(Generic[M]):
    """Symmetric serialize / deserialize between torch modules and safetensors.

    The canonical safetensor format uses:
      - ``gamma`` / ``beta`` for LayerNorm parameters
      - ``(in_features, out_features)`` layout for linear weights  (column-major)
      - flat ``(num_features, num_features)`` layout for attention projection weights

    Torch uses:
      - ``weight`` / ``bias`` for LayerNorm
      - ``(out_features, in_features)`` layout for ``nn.Linear``  (row-major)
    """

    def __init__(self, module: M):
        self.module = module
        self.map_to_torch: Dict[str, str] = {}
        self.map_to_safetensor: Dict[str, str] = {}
        self.linear_weight_keys: set[str] = set()

        for name, child in module.named_modules():
            prefix = name
            if isinstance(child, nn.LayerNorm):
                for st_suffix, torch_suffix in _LAYER_NORM_TO_TORCH.items():
                    st_key = f"{prefix}.{st_suffix}" if prefix else st_suffix
                    torch_key = f"{prefix}.{torch_suffix}" if prefix else torch_suffix
                    self._add(st_key, torch_key)
            elif isinstance(child, nn.Linear):
                key = f"{prefix}.weight" if prefix else "weight"
                self.linear_weight_keys.add(key)

    def _add(self, safetensor_key: str, torch_key: str):
        self.map_to_torch[safetensor_key] = torch_key
        self.map_to_safetensor[torch_key] = safetensor_key

    # -- serialize (torch → safetensor) --

    def _serialize_common(self) -> Dict[str, torch.Tensor]:
        state = self.module.state_dict()
        # Transpose linear weights: torch (out, in) → canonical (in, out)
        for key in self.linear_weight_keys:
            if key in state and state[key].ndim == 2:
                state[key] = state[key].t().contiguous()
        # Rename torch keys → canonical safetensor keys
        remap(state, self.map_to_safetensor, [])
        return dict(sorted(state.items()))

    def serialize(self) -> bytes:
        return save(self._serialize_common())

    def serialize_to_file(self, filename: Union[str, os.PathLike]) -> None:
        return save_file(self._serialize_common(), filename)

    # -- deserialize (safetensor → torch) --

    def _deserialize_common(self, tensors: Dict[str, torch.Tensor]) -> M:
        # Rename canonical keys → torch keys
        remap(tensors, self.map_to_torch, [])
        # Transpose linear weights: canonical (in, out) → torch (out, in)
        for key in self.linear_weight_keys:
            if key in tensors and tensors[key].ndim == 2:
                tensors[key] = tensors[key].t().contiguous()
        self.module.load_state_dict(tensors)
        return self.module

    def deserialize(self, data: bytes) -> M:
        return self._deserialize_common(load(data))

    def deserialize_from_file(self, filename: Union[str, os.PathLike]) -> M:
        return self._deserialize_common(load_file(filename))


def load_safetensor_into(module: nn.Module, data: bytes) -> None:
    """Load safetensors bytes into a torch module (convenience wrapper)."""
    Serde(module).deserialize(data)


class Serializable:
    """Mixin for nn.Module subclasses that adds safetensors serialization.

    Usage::

        class MyModel(Serializable, nn.Module):
            ...

        model.save("weights.safetensors")
        model = MyModel.load("weights.safetensors", ...)
    """

    def save(self, filename: Union[str, os.PathLike]) -> None:
        Serde(self).serialize_to_file(filename)

    def save_bytes(self) -> bytes:
        return Serde(self).serialize()

    @classmethod
    def load(cls, filename: Union[str, os.PathLike], *args, **kwargs):
        module = cls(*args, **kwargs)
        Serde(module).deserialize_from_file(filename)
        return module

    @classmethod
    def load_bytes(cls, data: bytes, *args, **kwargs):
        module = cls(*args, **kwargs)
        Serde(module).deserialize(data)
        return module
