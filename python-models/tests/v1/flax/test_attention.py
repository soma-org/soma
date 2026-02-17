from jax import numpy as jnp
from flax import nnx
from arrgen import (
    constant_array,
    normal_array,
)
from safetensors.numpy import save
from soma_models.v1.flax.serde import Serde
from soma_models.v1.flax.modules.attention import MultiHeadAttention, apply_rope


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
    outputs = module(inputs, positions=positions)
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


def test_v1_rope_multi_position():
    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_dim = 4
    max_wavelength = 10_000
    scale_factor = 1.0
    seed = 70

    inputs = jnp.array(
        normal_array(seed, [batch_size, seq_len, num_heads, head_dim], 0.0, 1.0)
    )
    positions = jnp.arange(0, seq_len).reshape(1, -1).repeat(batch_size, axis=0)
    outputs = apply_rope(inputs, positions, head_dim, max_wavelength, scale_factor)

    expected = jnp.array(
        [
            [
                [[0.47776070, -0.48515794, 0.26153922, -2.41498089],
                 [0.99376506, 0.90728259, -1.43244362, 0.09139032]],
                [[-0.06775475, 0.15760149, -0.07331341, 1.00924647],
                 [0.46687761, 1.30162942, -0.95026934, -0.30788255]],
                [[-0.76642752, -0.87616366, 2.00954461, -0.66655433],
                 [-0.53347385, 0.33248445, -0.25528368, 1.70450819]],
                [[0.68225688, 0.77837843, -1.03880394, -0.73422980],
                 [0.94099069, -0.55226529, -0.84172773, -0.39250299]],
            ],
            [
                [[0.53636390, 1.04667366, -0.62744135, -1.57136846],
                 [0.56636423, 0.85205758, -2.16002727, -0.19352338]],
                [[1.11170232, 1.09174263, -0.25253245, -0.36698121],
                 [1.58594680, -1.20130229, -0.42448375, -0.20691611]],
                [[0.38839683, 0.45338595, -0.52934897, -0.38279817],
                 [-0.89073908, 0.54021513, -1.11969602, -1.80852568]],
                [[-0.01806840, -0.24985033, -0.84394234, -0.19104014],
                 [-0.67495888, 1.27409673, -0.01402950, -1.07746637]],
            ],
        ]
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"


def test_v1_attention_multi_batch():
    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_dim = 2

    seed = 80
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
    outputs = module(inputs, positions=positions)
    expected = jnp.array(
        [
            [
                [6.06968307, -1.03515446, 1.70626497, -1.02048159],
                [5.82955551, 1.31834793, 1.28906751, -5.05607700],
                [-0.63638550, -1.46703827, -1.53339362, -0.55266494],
                [-2.60889316, 3.90301847, -0.06124383, 0.83532411],
            ],
            [
                [5.63261747, 3.12328148, 3.08034492, -0.17959267],
                [1.92174339, 4.35927916, 2.09976912, -0.74388230],
                [-6.87539053, -3.64232492, -5.18859148, 0.58245522],
                [-10.21397591, -1.89333570, -5.76631403, 0.58711916],
            ],
        ],
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"
