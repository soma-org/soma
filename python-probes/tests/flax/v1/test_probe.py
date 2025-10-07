from jax import numpy as jnp
from flax import nnx
from arrgen import (
    constant_array,
    normal_array,
)
from safetensors.numpy import save
from soma_probes.flax.serde import Serde
from soma_probes.flax.v1.probe import Probe, ProbeConfig


def test_v1_probe():
    seed = 42

    batch_size = 2
    seq_len = 1
    num_heads = 1
    head_dim = 2
    num_layers = 1
    max_wavelength = 10_000
    scale_factor = 1.0
    embedding_dim = head_dim * num_heads
    hidden_dim = embedding_dim * 2
    vocab_size = 1
    max_seq_len = 10

    serde = Serde(
        Probe(
            ProbeConfig(
                dropout_rate=0.0,
                embedding_dim=embedding_dim,
                pwff_hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                vocab_size=vocab_size,
                max_wavelength=max_wavelength,
                max_seq_len=max_seq_len,
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

    probe_tensors = {
        "mask_token": normal_array(
            seed + 100, [1, 1, embedding_dim], mean=0.0, std_dev=1.0
        ),
        "final_norm.gamma": normal_array(
            seed + 101, [embedding_dim], mean=0.0, std_dev=1.0
        ),
        "final_norm.beta": normal_array(
            seed + 102, [embedding_dim], mean=0.0, std_dev=1.0
        ),
        "predictor.weight": normal_array(
            seed + 103, [embedding_dim, vocab_size], mean=0.0, std_dev=1.0
        ),
        "predictor.bias": normal_array(seed + 104, [vocab_size], mean=0.0, std_dev=1.0),
    }

    generated_tensors.update(probe_tensors)

    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()
    inputs = jnp.array(constant_array([batch_size, seq_len, embedding_dim], 1.0))
    positions = jnp.array([[1], [1]])
    outputs = module(inputs, positions=positions)
    expected = jnp.array(
        [[1.01745927], [1.01745927]],
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"
