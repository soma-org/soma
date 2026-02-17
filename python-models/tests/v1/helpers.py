"""Shared test helpers for building deterministic model weights."""

from arrgen import normal_array


def small_config():
    return dict(
        dropout_rate=0.0,
        embedding_dim=16,
        pwff_hidden_dim=32,
        num_layers=2,
        num_heads=4,
        vocab_size=264,
    )


def build_model_weights(seed, num_layers, embedding_dim, hidden_dim, vocab_size):
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
    generated_tensors["final_norm.gamma"] = normal_array(
        seed + 100, [embedding_dim], mean=0.0, std_dev=1.0,
    )
    generated_tensors["final_norm.beta"] = normal_array(
        seed + 200, [embedding_dim], mean=0.0, std_dev=1.0,
    )
    generated_tensors["embedding.weight"] = normal_array(
        seed + 250, [vocab_size, embedding_dim], mean=0.0, std_dev=1.0,
    )
    generated_tensors["predictor.weight"] = normal_array(
        seed + 300, [embedding_dim, vocab_size], mean=0.0, std_dev=1.0,
    )
    generated_tensors["predictor.bias"] = normal_array(
        seed + 400, [vocab_size], mean=0.0, std_dev=1.0,
    )
    return generated_tensors
