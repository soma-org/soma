from jax import numpy as jnp
from flax import nnx
from arrgen import (
    uniform_array,
    normal_array,
    constant_array,
)
from safetensors.numpy import save
from soma_probes.flax.serde import Serde


class LayerNormModule(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.layer_norm = nnx.LayerNorm(
            num_features=4,
            epsilon=1e-5,
            use_fast_variance=False,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.layer_norm(x)
        return x


def test_layer_norm_ones():
    seed = 42
    serde = Serde(LayerNormModule(rngs=nnx.Rngs(0)))
    generated_tensors = {
        "layer_norm.gamma": normal_array(seed, [4], mean=0.0, std_dev=1.0),
        "layer_norm.beta": normal_array(seed + 1, [4], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()

    expected = jnp.array([0.26803425, -0.30034754, -0.18579677, -0.37248048])
    input = jnp.array(constant_array([4], 1.0))
    output = module(input)

    assert jnp.allclose(output, expected), "Arrays are not close enough!"


def test_layer_norm_uniform():
    seed = 44
    serde = Serde(LayerNormModule(rngs=nnx.Rngs(0)))
    generated_tensors = {
        "layer_norm.gamma": normal_array(seed, [4], mean=0.0, std_dev=1.0),
        "layer_norm.beta": normal_array(seed + 1, [4], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()

    expected = jnp.array([-0.74536324, -2.98460746, -0.31756663, -0.38157958])
    input = jnp.array(uniform_array(seed + 2, [4], 0.0, 1.0))
    output = module(input)

    assert jnp.allclose(output, expected), "Arrays are not close enough!"
