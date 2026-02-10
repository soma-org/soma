from jax import numpy as jnp
from flax import nnx
from arrgen import (
    normal_array,
)
from safetensors.numpy import save
from soma_models.flax.serde import Serde
from soma_models.flax.v1.probe import Probe, ProbeConfig


def test_v1_probe():
    seed = 42

    batch_size = 1
    vocab_size = 4
    seq_len = 4
    num_heads = 2
    head_dim = 2
    num_layers = 2
    embedding_dim = head_dim * num_heads
    hidden_dim = embedding_dim * 2

    serde = Serde(
        Probe(
            ProbeConfig(
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
            f"encoder.layers.{layer}.norm_1.gamma": normal_array(
                lseed + 1,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.norm_1.beta": normal_array(
                lseed + 2,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.query.weight": normal_array(
                lseed + 3,
                [embedding_dim, embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.query.bias": normal_array(
                lseed + 4,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.key.weight": normal_array(
                lseed + 5,
                [embedding_dim, embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.key.bias": normal_array(
                lseed + 6,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.value.weight": normal_array(
                lseed + 7,
                [embedding_dim, embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.value.bias": normal_array(
                lseed + 8,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.output.weight": normal_array(
                lseed + 9,
                [embedding_dim, embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.output.bias": normal_array(
                lseed + 10,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.norm_2.gamma": normal_array(
                lseed + 11,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.norm_2.beta": normal_array(
                lseed + 12,
                [embedding_dim],
                mean=0.0,
                std_dev=1.0,
            ),
            f"encoder.layers.{layer}.pwff.linear_inner.weight": normal_array(
                lseed + 13, [embedding_dim, hidden_dim], mean=0.0, std_dev=1.0
            ),
            f"encoder.layers.{layer}.pwff.linear_inner.bias": normal_array(
                lseed + 14, [hidden_dim], mean=0.0, std_dev=1.0
            ),
            f"encoder.layers.{layer}.pwff.linear_outer.weight": normal_array(
                lseed + 15, [hidden_dim, embedding_dim], mean=0.0, std_dev=1.0
            ),
            f"encoder.layers.{layer}.pwff.linear_outer.bias": normal_array(
                lseed + 16, [embedding_dim], mean=0.0, std_dev=1.0
            ),
        }
        generated_tensors.update(layer_tensors)
    last_tensors = {
        "final_norm.gamma": normal_array(
            seed + 100,
            [embedding_dim],
            mean=0.0,
            std_dev=1.0,
        ),
        "final_norm.beta": normal_array(
            seed + 200, [embedding_dim], mean=0.0, std_dev=1.0
        ),
        "embed.weight": normal_array(
            seed + 250, [vocab_size, embedding_dim], mean=0.0, std_dev=1.0
        ),
        "predictor.weight": normal_array(
            seed + 300, [embedding_dim, vocab_size], mean=0.0, std_dev=1.0
        ),
        "predictor.bias": normal_array(
            seed + 400, [embedding_dim], mean=0.0, std_dev=1.0
        ),
    }
    generated_tensors.update(last_tensors)

    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()
    inputs = jnp.array(
        normal_array(seed + 100, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    positions = jnp.arange(0, seq_len).reshape(1, -1).repeat(batch_size, axis=0)
    mask = nnx.make_causal_mask(positions, dtype=bool)
    outputs = module(inputs, positions=positions, attn_mask=mask)

    expected = jnp.array(
        [
            [
                [1.98016202, 0.81204247, 1.52166390, 2.38146710],
                [1.77488256, 0.79269153, 1.61762166, 2.30634165],
                [1.94405246, 0.93004227, 1.48189783, 2.45317531],
                [1.45131588, 0.90625042, 1.68207979, 2.26099873],
            ]
        ],
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"
