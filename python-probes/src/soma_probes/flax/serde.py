import os
from typing import Dict, Union
from soma_probes.utils import remap, flatten_dict, unflatten_dict
from safetensors.flax import save_file, load_file, save, load, Array
from flax import nnx


class Serde:
    def __init__(self, module: nnx.Module):
        self.module = module
        self.map_to_flax: Dict[str, str] = {}
        self.map_to_safetensor: Dict[str, str] = {}
        for path, submodule in module.iter_modules():
            flat_path = ".".join(path)
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

    def _serialize_common(self) -> Dict:
        state = nnx.state(self.module)
        state_dict = nnx.to_pure_dict(state)
        flat_state_dict = flatten_dict(state_dict)
        remap(flat_state_dict, self.map_to_safetensor)
        return flat_state_dict

    def serialize(self) -> bytes:
        state_dict = self._serialize_common()
        return save(state_dict)

    def serialize_to_file(self, filename: Union[str, os.PathLike]) -> None:
        state_dict = self._serialize_common()
        return save_file(state_dict, filename)

    def _deserialize_common(self, safetensor_dict: Dict[str, Array]) -> nnx.Module:
        remap(safetensor_dict, self.map_to_flax)
        nested_dict = unflatten_dict(safetensor_dict)
        state = nnx.state(self.module)
        nnx.replace_by_pure_dict(state, nested_dict)
        nnx.update(self.module, state)
        return self.module

    def deserialize(self, data: bytes) -> nnx.Module:
        safetensor_dict = load(data)
        return self._deserialize_common(safetensor_dict)

    def deserialize_from_file(self, filename: Union[str, os.PathLike]) -> nnx.Module:
        safetensor_dict = load_file(filename)
        return self._deserialize_common(safetensor_dict)
