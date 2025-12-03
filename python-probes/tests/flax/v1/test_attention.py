from jax import numpy as jnp
from flax import nnx
from arrgen import (
    constant_array,
    normal_array,
)
from safetensors.numpy import save
from soma_probes.flax.serde import Serde
from soma_probes.flax.v1.modules.attention import (
    MultiHeadAttention,
)


class MhaModule(nnx.Module):
    def __init__(self, num_heads, head_dim, rngs: nnx.Rngs):
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            num_features=num_heads * head_dim,
            rngs=nnx.Rngs(0),
        )

    def __call__(self, x):
        x = self.mha(inputs=x)
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
    outputs = module(inputs)
    expected = jnp.array(
        [
            [
                [0.02807204, 2.13409519, 5.45623493, 13.48927689],
                [3.20324492, 0.78216082, 0.03813785, 1.87561870],
                [3.04302406, 0.51064676, -0.37436891, 0.42570090],
                [3.08121729, 0.57106179, -0.28497183, 0.74396586],
            ]
        ],
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"
