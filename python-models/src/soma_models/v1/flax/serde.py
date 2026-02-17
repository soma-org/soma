import os
from typing import Dict, Union, List, TypeVar, Generic, Tuple
from soma_models.utils import remap, flatten_dict, unflatten_dict
from safetensors.flax import save_file, load_file, save, load, Array
from flax import nnx

M = TypeVar("M", bound=nnx.Module)

# (safetensor_suffix, flax_suffix) pairs per module type
_LAYER_MAPPINGS: Dict[type, List[Tuple[str, str]]] = {
    nnx.Embed: [("weight", "embedding")],
    nnx.Linear: [("weight", "kernel")],
    nnx.LayerNorm: [("gamma", "scale"), ("beta", "bias")],
}

_MHA_MAPPINGS: List[Tuple[str, str]] = [
    ("query.weight", "query.kernel"),
    ("key.weight", "key.kernel"),
    ("value.weight", "value.kernel"),
    ("output.weight", "out.kernel"),
    ("output.bias", "out.bias"),
]


class Serde(Generic[M]):
    def __init__(self, module: M):
        self.module = module
        self.attention_shapes: Dict[str, Tuple[int, int]] = {}
        self.map_to_flax: Dict[str, str] = {}
        self.map_to_safetensor: Dict[str, str] = {}
        self.remove: List[str] = []

        for path, submodule in nnx.iter_modules(module):
            prefix = ".".join(str(item) for item in path)

            if isinstance(submodule, nnx.Dropout):
                continue

            from soma_models.v1.flax.modules.attention import MultiHeadAttention
            if isinstance(submodule, MultiHeadAttention):
                self.attention_shapes[prefix] = (
                    submodule.num_heads,
                    submodule.head_dim,
                )
                for st_suffix, flax_suffix in _MHA_MAPPINGS:
                    self._add(f"{prefix}.{st_suffix}", f"{prefix}.{flax_suffix}")
                continue

            mappings = _LAYER_MAPPINGS.get(type(submodule))
            if mappings:
                for st_suffix, flax_suffix in mappings:
                    self._add(f"{prefix}.{st_suffix}", f"{prefix}.{flax_suffix}")

    def _add(self, safetensor_key: str, flax_key: str):
        self.map_to_flax[safetensor_key] = flax_key
        self.map_to_safetensor[flax_key] = safetensor_key

    def _reshape_attention(
        self,
        params: Dict[str, Array],
        name: str,
        num_heads: int,
        head_dim: int,
        *,
        to_flax: bool,
    ):
        """Reshape attention parameters between flat (safetensor) and split-head (flax) layouts."""
        num_features = num_heads * head_dim
        flat = (num_features, num_features)
        # Key names differ by direction: safetensor uses "weight"/"output", flax uses "kernel"/"out"
        if to_flax:
            qkv_key = "weight"
            out_w_key = f"{name}.output.weight"
            out_b_key = f"{name}.output.bias"
        else:
            qkv_key = "kernel"
            out_w_key = f"{name}.out.kernel"
            out_b_key = f"{name}.out.bias"

        for component in ["query", "key", "value"]:
            w = f"{name}.{component}.{qkv_key}"
            if w in params:
                params[w] = params[w].reshape(
                    (num_features, num_heads, head_dim) if to_flax else flat
                )
            b = f"{name}.{component}.bias"
            if b in params:
                params[b] = params[b].reshape(
                    (num_heads, head_dim) if to_flax else (num_features,)
                )

        if out_w_key in params:
            params[out_w_key] = params[out_w_key].reshape(
                (num_heads, head_dim, num_features) if to_flax else flat
            )

    def _serialize_common(self) -> Dict:
        state = nnx.state(self.module)
        state_dict = nnx.to_pure_dict(state)
        flat_state_dict = flatten_dict(state_dict)
        # Remove all RNG stream state (rngs.count / rngs.key) from any module
        rng_keys = [k for k in flat_state_dict if k.endswith(".rngs.count") or k.endswith(".rngs.key")]
        for k in rng_keys:
            del flat_state_dict[k]
        for name, (num_heads, head_dim) in self.attention_shapes.items():
            self._reshape_attention(
                flat_state_dict, name, num_heads, head_dim, to_flax=False
            )
        remap(flat_state_dict, self.map_to_safetensor, self.remove)
        return dict(sorted(flat_state_dict.items()))

    def serialize(self) -> bytes:
        return save(self._serialize_common())

    def serialize_to_file(self, filename: Union[str, os.PathLike]) -> None:
        return save_file(self._serialize_common(), filename)

    def _deserialize_common(self, safetensor_dict: Dict[str, Array]) -> M:
        for name, (num_heads, head_dim) in self.attention_shapes.items():
            self._reshape_attention(
                safetensor_dict, name, num_heads, head_dim, to_flax=True
            )
        remap(safetensor_dict, self.map_to_flax, [])
        nested_dict = unflatten_dict(safetensor_dict)
        state = nnx.state(self.module)
        nnx.replace_by_pure_dict(state, nested_dict)
        nnx.update(self.module, state)
        return self.module

    def deserialize(self, data: bytes) -> M:
        return self._deserialize_common(load(data))

    def deserialize_from_file(self, filename: Union[str, os.PathLike]) -> M:
        return self._deserialize_common(load_file(filename))


class Serializable:
    """Mixin for nnx.Module subclasses that adds safetensors serialization."""

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
