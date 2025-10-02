from jax import numpy as jnp
from flax import nnx
from arrgen import (
    constant_array,
    normal_array,
)
from safetensors.numpy import save
from soma_probes.flax.serde import Serde
from soma_probes.flax.v1.modules.attention import (
    apply_rope,
    MultiHeadAttention,
)


def test_v1_rope_ones():
    batch_size = 1
    seq_len = 1
    num_heads = 1
    head_dim = 2
    max_wavelength = 10_000
    scale_factor = 1.0

    inputs = jnp.array(constant_array([batch_size, seq_len, num_heads, head_dim], 1.0))
    positions = jnp.array([[1]])
    outputs = apply_rope(inputs, positions, head_dim, max_wavelength, scale_factor)

    expected = jnp.array([[[[-0.30116868, 1.38177323]]]])
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"


class MhaModule(nnx.Module):
    def __init__(self, num_heads, head_dim, rngs: nnx.Rngs):
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            num_features=num_heads * head_dim,
            rngs=nnx.Rngs(0),
        )

    def __call__(self, x, positions):
        x = self.mha(inputs=x, positions=positions)
        return x


def test_v1_attention():
    batch_size = 1
    seq_len = 1
    num_heads = 1
    head_dim = 2
    max_wavelength = 10_000.0
    scale_factor = 1.0

    seed = 42
    serde = Serde(MhaModule(num_heads, head_dim, rngs=nnx.Rngs(0)))
    generated_tensors = {
        "mha.query.weight": normal_array(
            seed + 1,
            [num_heads * head_dim, num_heads * head_dim],
            mean=0.0,
            std_dev=1.0,
        ),
        "mha.query.bias": normal_array(
            seed + 2, [num_heads * head_dim], mean=0.0, std_dev=1.0
        ),
        "mha.key.weight": normal_array(
            seed + 3,
            [num_heads * head_dim, num_heads * head_dim],
            mean=0.0,
            std_dev=1.0,
        ),
        "mha.key.bias": normal_array(
            seed + 4, [num_heads * head_dim], mean=0.0, std_dev=1.0
        ),
        "mha.value.weight": normal_array(
            seed + 5,
            [num_heads * head_dim, num_heads * head_dim],
            mean=0.0,
            std_dev=1.0,
        ),
        "mha.value.bias": normal_array(
            seed + 6, [num_heads * head_dim], mean=0.0, std_dev=1.0
        ),
        "mha.output.weight": normal_array(
            seed + 7,
            [num_heads * head_dim, num_heads * head_dim],
            mean=0.0,
            std_dev=1.0,
        ),
        "mha.output.bias": normal_array(
            seed + 8, [num_heads * head_dim], mean=0.0, std_dev=1.0
        ),
    }
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()

    inputs = jnp.array(constant_array([batch_size, seq_len, num_heads * head_dim], 1.0))
    positions = jnp.array([[1]])
    outputs = module(inputs, positions)
    expected = jnp.array([[[-1.93152618, -0.17658120]]])
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"
