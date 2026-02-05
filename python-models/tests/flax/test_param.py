from jax import numpy as jnp
from flax import nnx
from arrgen import (
    normal_array,
)
from safetensors.numpy import save
from soma_models.flax.serde import Serde


class ParamModule1D(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.param = nnx.Param(nnx.initializers.zeros(rngs.params(), (4)))

    def __call__(self):
        return self.param


class ParamModule3D(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.param = nnx.Param(nnx.initializers.zeros(rngs.params(), (1, 1, 4)))

    def __call__(self):
        return self.param


def test_1d_param():
    seed = 42
    embedding_dim = 4
    serde = Serde(ParamModule1D(rngs=nnx.Rngs(0)))
    generated_tensors = {
        "param": normal_array(seed, [embedding_dim], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()

    expected = jnp.array(
        [0.06942791, 0.13293812, 0.26257637, -0.22530088],
    )
    output = module()

    assert jnp.allclose(output, expected), "Arrays are not close enough!"


def test_3d_param():
    seed = 42
    embedding_dim = 4
    serde = Serde(ParamModule1D(rngs=nnx.Rngs(0)))
    generated_tensors = {
        "param": normal_array(seed, [1, 1, embedding_dim], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()

    expected = jnp.array(
        [[[0.06942791, 0.13293812, 0.26257637, -0.22530088]]],
    )
    output = module()

    assert jnp.allclose(output, expected), "Arrays are not close enough!"
