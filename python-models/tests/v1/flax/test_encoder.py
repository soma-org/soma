from jax import numpy as jnp
from flax import nnx
from arrgen import (
    normal_array,
)
from safetensors.numpy import save
from soma_models.v1.flax.serde import Serde
from soma_models.v1.flax.modules.encoder import Encoder, EncoderConfig


def test_v1_attention():
    seed = 42
    batch_size = 1
    seq_len = 4
    num_heads = 2
    head_dim = 2
    num_layers = 2
    embedding_dim = head_dim * num_heads
    hidden_dim = embedding_dim * 2
    serde = Serde(
        Encoder(
            EncoderConfig(
                dropout_rate=0.0,
                embedding_dim=embedding_dim,
                pwff_hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
            ),
            rngs=nnx.Rngs(0),
        )
    )
    generated_tensors = {}
    for layer in range(num_layers):
        lseed = seed + layer
        layer_tensors = {
            f"layers.{layer}.norm_1.gamma": normal_array(
                lseed + 1,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.norm_1.beta": normal_array(
                lseed + 2,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.attention.query.weight": normal_array(
                lseed + 3,
                [embedding_dim, embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.attention.query.bias": normal_array(
                lseed + 4,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.attention.key.weight": normal_array(
                lseed + 5,
                [embedding_dim, embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.attention.key.bias": normal_array(
                lseed + 6,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.attention.value.weight": normal_array(
                lseed + 7,
                [embedding_dim, embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.attention.value.bias": normal_array(
                lseed + 8,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.attention.output.weight": normal_array(
                lseed + 9,
                [embedding_dim, embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.attention.output.bias": normal_array(
                lseed + 10,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.norm_2.gamma": normal_array(
                lseed + 11,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.norm_2.beta": normal_array(
                lseed + 12,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"layers.{layer}.pwff.linear_inner.weight": normal_array(
                lseed + 13, [embedding_dim, hidden_dim], mean=0.0, std_dev=1.0
            ),
            f"layers.{layer}.pwff.linear_inner.bias": normal_array(
                lseed + 14, [hidden_dim], mean=0.0, std_dev=1.0
            ),
            f"layers.{layer}.pwff.linear_outer.weight": normal_array(
                lseed + 15, [hidden_dim, embedding_dim], mean=0.0, std_dev=1.0
            ),
            f"layers.{layer}.pwff.linear_outer.bias": normal_array(
                lseed + 16, [embedding_dim], mean=0.0, std_dev=1.0
            ),
        }
        generated_tensors.update(layer_tensors)
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()
    inputs = jnp.array(
        normal_array(seed + 100, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    positions = jnp.arange(0, seq_len).reshape(1, -1).repeat(batch_size, axis=0)
    outputs = module(inputs, positions=positions)
    print(outputs)
    expected = jnp.array(
        [
            [
                [8.33097267, 3.23748636, 12.93889809, -4.79724693],
                [5.95732164, 2.65965629, 11.97813892, -3.51363301],
                [8.32810211, 4.93732452, 12.48622322, -4.29308748],
                [3.45333171, 3.12663937, 10.79409981, -2.80449820],
            ]
        ],
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"


def _build_encoder_tensors(seed, num_layers, embedding_dim, hidden_dim):
    generated_tensors = {}
    for layer in range(num_layers):
        lseed = seed + layer
        layer_tensors = {
            f"layers.{layer}.norm_1.gamma": normal_array(
                lseed + 1, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.norm_1.beta": normal_array(
                lseed + 2, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.attention.query.weight": normal_array(
                lseed + 3, [embedding_dim, embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.attention.query.bias": normal_array(
                lseed + 4, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.attention.key.weight": normal_array(
                lseed + 5, [embedding_dim, embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.attention.key.bias": normal_array(
                lseed + 6, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.attention.value.weight": normal_array(
                lseed + 7, [embedding_dim, embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.attention.value.bias": normal_array(
                lseed + 8, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.attention.output.weight": normal_array(
                lseed + 9, [embedding_dim, embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.attention.output.bias": normal_array(
                lseed + 10, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.norm_2.gamma": normal_array(
                lseed + 11, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.norm_2.beta": normal_array(
                lseed + 12, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.pwff.linear_inner.weight": normal_array(
                lseed + 13, [embedding_dim, hidden_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.pwff.linear_inner.bias": normal_array(
                lseed + 14, [hidden_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.pwff.linear_outer.weight": normal_array(
                lseed + 15, [hidden_dim, embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"layers.{layer}.pwff.linear_outer.bias": normal_array(
                lseed + 16, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
        }
        generated_tensors.update(layer_tensors)
    return generated_tensors


def test_v1_encoder_normal_input():
    seed = 90
    batch_size = 1
    seq_len = 4
    num_heads = 2
    head_dim = 2
    num_layers = 2
    embedding_dim = head_dim * num_heads
    hidden_dim = embedding_dim * 2
    serde = Serde(
        Encoder(
            EncoderConfig(
                dropout_rate=0.0,
                embedding_dim=embedding_dim,
                pwff_hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
            ),
            rngs=nnx.Rngs(0),
        )
    )
    generated_tensors = _build_encoder_tensors(seed, num_layers, embedding_dim, hidden_dim)
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()
    inputs = jnp.array(
        normal_array(seed + 200, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    positions = jnp.arange(0, seq_len).reshape(1, -1).repeat(batch_size, axis=0)
    outputs = module(inputs, positions=positions)
    expected = jnp.array(
        [[
            [-3.17372561, 4.01577044, 19.12659073, 4.34993553],
            [-1.27379203, 5.61034679, 18.69484711, 4.15108776],
            [-0.34562051, 5.53026056, 20.70455742, 2.78899574],
            [-13.14624596, 5.65161657, 19.76067924, 1.81537509],
        ]],
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"


def test_v1_encoder_multi_batch():
    seed = 100
    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_dim = 2
    num_layers = 1
    embedding_dim = head_dim * num_heads
    hidden_dim = embedding_dim * 2
    serde = Serde(
        Encoder(
            EncoderConfig(
                dropout_rate=0.0,
                embedding_dim=embedding_dim,
                pwff_hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
            ),
            rngs=nnx.Rngs(0),
        )
    )
    generated_tensors = _build_encoder_tensors(seed, num_layers, embedding_dim, hidden_dim)
    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()
    inputs = jnp.array(
        normal_array(seed + 200, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    positions = jnp.arange(0, seq_len).reshape(1, -1).repeat(batch_size, axis=0)
    outputs = module(inputs, positions=positions)
    expected = jnp.array(
        [
            [
                [-5.37061596, 2.96268249, 1.01008272, -5.05764151],
                [-2.89370632, -0.09757829, -0.53730059, -5.81040239],
                [-2.18361926, -5.22197628, 1.21352792, -10.84550667],
                [-4.18923759, -2.77440166, 0.54678917, -6.34802437],
            ],
            [
                [2.99903393, -8.75441551, -1.97783256, -6.19660091],
                [1.16564727, -5.46568251, -0.88867092, -5.41934681],
                [-2.09243917, 1.01158941, 0.81291533, -4.70484543],
                [-3.40283728, -3.52438331, 1.75270081, -7.97897339],
            ],
        ],
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"
