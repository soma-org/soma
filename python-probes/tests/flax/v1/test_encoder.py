from jax import numpy as jnp
from flax import nnx
from arrgen import (
    constant_array,
    normal_array,
)
from safetensors.numpy import save
from soma_probes.flax.serde import Serde
from soma_probes.flax.v1.modules.encoder import Encoder, EncoderConfig


def test_v1_attention():
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
    serde = Serde(
        Encoder(
            EncoderConfig(
                dropout_rate=0.0,
                embedding_dim=embedding_dim,
                pwff_hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_wavelength=max_wavelength,
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
    inputs = jnp.array(constant_array([batch_size, seq_len, embedding_dim], 1.0))
    positions = jnp.array([[1]])
    outputs = module(inputs, positions=positions)
    print(outputs)
    expected = jnp.array(
        [[[0.6717827, 2.158453]], [[0.6717827, 2.158453]]],
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"
