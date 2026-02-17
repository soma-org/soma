from jax import numpy as jnp
from flax import nnx
from arrgen import (
    uniform_array,
    normal_array,
    constant_array,
)
from safetensors.numpy import save
from soma_models.v1.flax.serde import Serde


class LinearModule(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            in_features=2,
            out_features=4,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.linear(x)
        return x


def test_linear_ones():
    seed = 42
    input_dim = 2
    output_dim = 4
    serde = Serde(LinearModule(rngs=nnx.Rngs(0)))
    generated_tensors = {
        "linear.weight": normal_array(
            seed + 1, [input_dim, output_dim], mean=0.0, std_dev=1.0
        ),
        "linear.bias": normal_array(seed, [output_dim], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()

    expected = jnp.array([-1.77364016, 1.29809809, -0.31307063, -1.68842816])
    input = jnp.array(constant_array([input_dim], 1.0))
    output = module(input)

    assert jnp.allclose(output, expected), "Arrays are not close enough!"


def test_linear_uniform():
    seed = 44
    input_dim = 2
    output_dim = 4
    serde = Serde(LinearModule(rngs=nnx.Rngs(0)))
    generated_tensors = {
        "linear.weight": normal_array(
            seed + 1, [input_dim, output_dim], mean=0.0, std_dev=1.0
        ),
        "linear.bias": normal_array(seed, [output_dim], mean=0.0, std_dev=1.0),
    }
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()

    expected = jnp.array([-0.53813028, -1.69855022, 0.92013592, 0.92915082])
    input = jnp.array(uniform_array(seed + 2, [input_dim], 0.0, 1.0))
    output = module(input)

    assert jnp.allclose(output, expected), "Arrays are not close enough!"
