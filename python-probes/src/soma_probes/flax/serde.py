import os
from typing import Dict, Union, List, TypeVar, Generic
from soma_probes.flax.v1.modules.attention import (
    MultiHeadAttention as MultiHeadAttentionV1,
)
from soma_probes.utils import remap, flatten_dict, unflatten_dict
from safetensors.flax import save_file, load_file, save, load, Array
from flax import nnx

M = TypeVar("M", bound=nnx.Module)


class Serde(Generic[M]):
    def __init__(self, module: M):
        self.module = module
        self.attention_shapes: Dict[str, tuple] = {}
        self.map_to_flax: Dict[str, str] = {}
        self.map_to_safetensor: Dict[str, str] = {}
        self.remove: List[str] = []
        for path, submodule in module.iter_modules():
            flat_path = ".".join(str(item) for item in path)
            match submodule:
                case nnx.Linear() as module:
                    self.map_to_flax[f"{flat_path}.weight"] = f"{flat_path}.kernel"
                    self.map_to_safetensor[f"{flat_path}.kernel"] = (
                        f"{flat_path}.weight"
                    )
                case nnx.LayerNorm() as module:
                    self.map_to_flax[f"{flat_path}.gamma"] = f"{flat_path}.scale"
                    self.map_to_flax[f"{flat_path}.beta"] = f"{flat_path}.bias"
                    self.map_to_safetensor[f"{flat_path}.scale"] = f"{flat_path}.gamma"
                    self.map_to_safetensor[f"{flat_path}.bias"] = f"{flat_path}.beta"
                case MultiHeadAttentionV1() as module:
                    self.attention_shapes[flat_path] = (
                        module.num_heads,
                        module.head_dim,
                    )
                    self.map_to_flax[f"{flat_path}.query.weight"] = (
                        f"{flat_path}.query.kernel"
                    )
                    self.map_to_safetensor[f"{flat_path}.query.kernel"] = (
                        f"{flat_path}.query.weight"
                    )
                    self.map_to_flax[f"{flat_path}.key.weight"] = (
                        f"{flat_path}.key.kernel"
                    )
                    self.map_to_safetensor[f"{flat_path}.key.kernel"] = (
                        f"{flat_path}.key.weight"
                    )
                    self.map_to_flax[f"{flat_path}.value.weight"] = (
                        f"{flat_path}.value.kernel"
                    )
                    self.map_to_safetensor[f"{flat_path}.value.kernel"] = (
                        f"{flat_path}.value.weight"
                    )
                    self.map_to_flax[f"{flat_path}.output.weight"] = (
                        f"{flat_path}.out.kernel"
                    )
                    self.map_to_safetensor[f"{flat_path}.out.kernel"] = (
                        f"{flat_path}.output.weight"
                    )
                    self.map_to_flax[f"{flat_path}.output.bias"] = (
                        f"{flat_path}.out.bias"
                    )
                    self.map_to_safetensor[f"{flat_path}.out.bias"] = (
                        f"{flat_path}.output.bias"
                    )
                case nnx.Dropout():
                    self.remove.append(f"{flat_path}.rngs.count")
                    self.remove.append(f"{flat_path}.rngs.key")
                    continue

    def _reshape_to_flax_attention(
        self, params: Dict[str, Array], name: str, num_heads: int, head_dim: int
    ):
        """Reshape target attention parameters to Flax MHA layout."""
        num_features = num_heads * head_dim
        # Reshape query, key, value weights: (num_features, num_features) -> (num_features, num_heads, head_dim)
        for component in ["query", "key", "value"]:
            weight_key = f"{name}.{component}.weight"
            if weight_key in params:
                params[weight_key] = params[weight_key].reshape(
                    num_features, num_heads, head_dim
                )
            bias_key = f"{name}.{component}.bias"
            if bias_key in params:
                params[bias_key] = params[bias_key].reshape(num_heads, head_dim)
        # Reshape output weight: (num_features, num_features) -> (num_heads, head_dim, num_features)
        out_weight_key = f"{name}.output.weight"
        if out_weight_key in params:
            params[out_weight_key] = params[out_weight_key].reshape(
                num_heads, head_dim, num_features
            )
        # Output bias is already (num_features,), no change needed
        out_bias_key = f"{name}.output.bias"
        if out_bias_key in params:
            assert params[out_bias_key].shape == (num_features,), (
                f"Expected shape ({num_features},) for {out_bias_key}"
            )

    def _reshape_from_flax_attention(
        self, params: Dict[str, Array], name: str, num_heads: int, head_dim: int
    ):
        """Reshape Flax MHA parameters to target linear layout."""
        num_features = num_heads * head_dim
        # Reshape query, key, value kernels: (num_features, num_heads, head_dim) -> (num_features, num_features)
        for component in ["query", "key", "value"]:
            kernel_key = f"{name}.{component}.kernel"
            if kernel_key in params:
                params[kernel_key] = params[kernel_key].reshape(
                    num_features, num_features
                )
            bias_key = f"{name}.{component}.bias"
            if bias_key in params:
                params[bias_key] = params[bias_key].reshape(num_features)
        # Reshape output kernel: (num_heads, head_dim, num_features) -> (num_features, num_features)
        out_kernel_key = f"{name}.out.kernel"
        if out_kernel_key in params:
            params[out_kernel_key] = params[out_kernel_key].reshape(
                num_features, num_features
            )
        # Output bias is already (num_features,), no change needed
        out_bias_key = f"{name}.out.bias"
        if out_bias_key in params:
            assert params[out_bias_key].shape == (num_features,), (
                f"Expected shape ({num_features},) for {out_bias_key}"
            )

    def _serialize_common(self) -> Dict:
        state = nnx.state(self.module)
        state_dict = nnx.to_pure_dict(state)
        flat_state_dict = flatten_dict(state_dict)
        for name, (num_heads, head_dim) in self.attention_shapes.items():
            self._reshape_from_flax_attention(
                flat_state_dict, name, num_heads, head_dim
            )
        remap(flat_state_dict, self.map_to_safetensor, self.remove)
        return flat_state_dict

    def serialize(self) -> bytes:
        state_dict = self._serialize_common()
        return save(state_dict)

    def serialize_to_file(self, filename: Union[str, os.PathLike]) -> None:
        state_dict = self._serialize_common()
        return save_file(state_dict, filename)

    def _deserialize_common(self, safetensor_dict: Dict[str, Array]) -> M:
        for name, (num_heads, head_dim) in self.attention_shapes.items():
            self._reshape_to_flax_attention(safetensor_dict, name, num_heads, head_dim)
        remap(safetensor_dict, self.map_to_flax, [])
        nested_dict = unflatten_dict(safetensor_dict)
        state = nnx.state(self.module)
        nnx.replace_by_pure_dict(state, nested_dict)
        nnx.update(self.module, state)
        return self.module

    def deserialize(self, data: bytes) -> M:
        safetensor_dict = load(data)
        return self._deserialize_common(safetensor_dict)

    def deserialize_from_file(self, filename: Union[str, os.PathLike]) -> M:
        safetensor_dict = load_file(filename)
        return self._deserialize_common(safetensor_dict)
