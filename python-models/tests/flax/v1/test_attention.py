from jax import numpy as jnp
from flax import nnx
from arrgen import (
    constant_array,
    normal_array,
)
from safetensors.numpy import save
from soma_models.flax.serde import Serde
from soma_models.flax.v1.modules.attention import MultiHeadAttention, apply_rope


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

    def __call__(self, x, positions, mask):
        x = self.mha(inputs=x, positions=positions, mask=mask)
        return x


def test_v1_attention():
    batch_size = 1
    seq_len = 4
    num_heads = 2
    head_dim = 2

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

    inputs = jnp.array(
        normal_array(seed + 9, [batch_size, seq_len, num_heads * head_dim], 0.0, 1.0)
    )
    positions = jnp.arange(0, seq_len).reshape(1, -1).repeat(batch_size, axis=0)
    mask = nnx.make_causal_mask(positions, dtype=bool)
    outputs = module(inputs, positions=positions, mask=mask)
    expected = jnp.array(
        [
            [
                [-3.59691596, -0.79137892, 2.59000635, 1.15417659],
                [-1.89609909, 0.90684074, 4.74483156, 9.88230705],
                [4.25082970, 2.23179317, 2.04289985, 9.17799950],
                [1.67264295, 1.73140073, 3.30692792, 10.11232185],
            ]
        ],
    )
    print(outputs)
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"
