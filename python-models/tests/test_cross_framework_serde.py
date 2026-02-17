"""Verify that safetensors produced by Flax and Torch are mutually loadable."""

import numpy as np
import torch
from flax import nnx
from safetensors.numpy import load as np_load

from soma_models.flax.v1.model import Model as FlaxModel, ModelConfig as FlaxModelConfig
from soma_models.flax.serde import Serde as FlaxSerde
from soma_models.torch.v1.model import Model as TorchModel, ModelConfig as TorchModelConfig
from soma_models.torch.serde import Serde as TorchSerde


def _small_config():
    return dict(
        dropout_rate=0.0,
        embedding_dim=64,
        pwff_hidden_dim=256,
        num_layers=1,
        num_heads=4,
        vocab_size=32,
    )


def test_flax_safetensor_loads_into_torch():
    """Serialize from Flax, deserialize into Torch — weights should match."""
    flax_model = FlaxModel(FlaxModelConfig(**_small_config()), rngs=nnx.Rngs(42))
    data = flax_model.save_bytes()

    torch_model = TorchModel(TorchModelConfig(**_small_config()))
    TorchSerde(torch_model).deserialize(data)

    # Re-serialize from torch and compare canonical keys
    torch_data = TorchSerde(torch_model).serialize()
    flax_tensors = np_load(data)
    torch_tensors = np_load(torch_data)
    assert sorted(flax_tensors.keys()) == sorted(torch_tensors.keys()), (
        f"Key mismatch: {sorted(flax_tensors.keys())} vs {sorted(torch_tensors.keys())}"
    )


def test_torch_safetensor_loads_into_flax():
    """Serialize from Torch, deserialize into Flax — should not raise."""
    torch.manual_seed(42)
    torch_model = TorchModel(TorchModelConfig(**_small_config()))
    data = torch_model.save_bytes()

    flax_model = FlaxModel(FlaxModelConfig(**_small_config()), rngs=nnx.Rngs(0))
    FlaxSerde(flax_model).deserialize(data)

    # Re-serialize from flax and verify values match
    flax_data = FlaxSerde(flax_model).serialize()
    torch_tensors = np_load(data)
    flax_tensors = np_load(flax_data)

    assert sorted(torch_tensors.keys()) == sorted(flax_tensors.keys())
    for key in torch_tensors:
        np.testing.assert_allclose(
            torch_tensors[key],
            flax_tensors[key],
            atol=1e-6,
            err_msg=f"Value mismatch at {key}",
        )
