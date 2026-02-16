from jax import numpy as jnp
from flax import nnx
from arrgen import (
    normal_array,
)
from safetensors.numpy import save
from soma_models.flax.serde import Serde
from soma_models.flax.v1.model import Model, ModelConfig


def test_v1_model():
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
        Model(
            ModelConfig(
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
        "embedding.weight": normal_array(
            seed + 250, [vocab_size, embedding_dim], mean=0.0, std_dev=1.0
        ),
        "predictor.weight": normal_array(
            seed + 300, [embedding_dim, vocab_size], mean=0.0, std_dev=1.0
        ),
        "predictor.bias": normal_array(
            seed + 400, [vocab_size], mean=0.0, std_dev=1.0
        ),
    }
    generated_tensors.update(last_tensors)

    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()
    tokens = jnp.array([[0, 1, 2, 3]])
    positions = jnp.arange(0, seq_len).reshape(1, -1).repeat(batch_size, axis=0)
    mask = nnx.make_causal_mask(positions, dtype=bool)
    outputs = module(tokens, positions=positions, attn_mask=mask)

    expected = jnp.array(
        [[
            [2.09210730, 0.69636524, 1.51327145, 2.31296515],
            [1.90634847, 0.86709338, 1.53078938, 2.40285897],
            [1.77797925, 0.66600311, 1.65812206, 2.19417143],
            [1.96223700, 0.77187395, 1.54670000, 2.34615612],
        ]],
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"


def test_v1_predict():
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
        Model(
            ModelConfig(
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
                lseed + 1, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.norm_1.beta": normal_array(
                lseed + 2, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.query.weight": normal_array(
                lseed + 3, [embedding_dim, embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.query.bias": normal_array(
                lseed + 4, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.key.weight": normal_array(
                lseed + 5, [embedding_dim, embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.key.bias": normal_array(
                lseed + 6, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.value.weight": normal_array(
                lseed + 7, [embedding_dim, embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.value.bias": normal_array(
                lseed + 8, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.output.weight": normal_array(
                lseed + 9, [embedding_dim, embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.attention.output.bias": normal_array(
                lseed + 10, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.norm_2.gamma": normal_array(
                lseed + 11, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.norm_2.beta": normal_array(
                lseed + 12, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.pwff.linear_inner.weight": normal_array(
                lseed + 13, [embedding_dim, hidden_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.pwff.linear_inner.bias": normal_array(
                lseed + 14, [hidden_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.pwff.linear_outer.weight": normal_array(
                lseed + 15, [hidden_dim, embedding_dim], mean=0.0, std_dev=1.0,
            ),
            f"encoder.layers.{layer}.pwff.linear_outer.bias": normal_array(
                lseed + 16, [embedding_dim], mean=0.0, std_dev=1.0,
            ),
        }
        generated_tensors.update(layer_tensors)
    last_tensors = {
        "final_norm.gamma": normal_array(
            seed + 100, [embedding_dim], mean=0.0, std_dev=1.0,
        ),
        "final_norm.beta": normal_array(
            seed + 200, [embedding_dim], mean=0.0, std_dev=1.0,
        ),
        "embedding.weight": normal_array(
            seed + 250, [vocab_size, embedding_dim], mean=0.0, std_dev=1.0,
        ),
        "predictor.weight": normal_array(
            seed + 300, [embedding_dim, vocab_size], mean=0.0, std_dev=1.0,
        ),
        "predictor.bias": normal_array(
            seed + 400, [vocab_size], mean=0.0, std_dev=1.0,
        ),
    }
    generated_tensors.update(last_tensors)

    serialized_tensors = save(generated_tensors)
    module = serde.deserialize(serialized_tensors)
    module.eval()
    tokens = jnp.array([[0, 1, 2, 3]])
    positions = jnp.arange(0, seq_len).reshape(1, -1).repeat(batch_size, axis=0)
    mask = nnx.make_causal_mask(positions, dtype=bool)
    encoded = module(tokens, positions=positions, attn_mask=mask)
    outputs = module.predict(encoded)

    expected = jnp.array(
        [[
            [1.89312398, -3.37666154, -5.06565857, -9.11204624],
            [1.40491748, -3.30115676, -5.32568741, -8.60974312],
            [1.71110404, -2.86263919, -4.95488262, -8.56918144],
            [1.63806129, -3.27318168, -5.17767763, -8.81518459],
        ]],
    )
    assert jnp.allclose(outputs, expected), "Arrays are not close enough!"
