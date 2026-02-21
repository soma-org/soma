"""Shared model weight utilities for Soma examples.

Provides constants and ``build_model_weights()`` for generating valid safetensors
weights that match the small-model config (embedding_dim=16, num_layers=2).
"""

from arrgen import normal_array

# Must match the --small-model config in the scoring service.
EMBEDDING_DIM = 16
PWFF_HIDDEN_DIM = 32
NUM_LAYERS = 2
NUM_HEADS = 4
VOCAB_SIZE = 264


def build_model_weights(seed: int) -> dict:
    """Generate valid safetensors weights for the small model config.

    Mirrors the Rust model architecture: embedding + transformer encoder +
    final layer norm + predictor.
    """
    tensors = {}
    for layer in range(NUM_LAYERS):
        lseed = seed + layer
        d, h = EMBEDDING_DIM, PWFF_HIDDEN_DIM
        tensors.update(
            {
                f"encoder.layers.{layer}.norm_1.gamma": normal_array(
                    lseed + 1, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.norm_1.beta": normal_array(
                    lseed + 2, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.query.weight": normal_array(
                    lseed + 3, [d, d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.query.bias": normal_array(
                    lseed + 4, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.key.weight": normal_array(
                    lseed + 5, [d, d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.key.bias": normal_array(
                    lseed + 6, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.value.weight": normal_array(
                    lseed + 7, [d, d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.value.bias": normal_array(
                    lseed + 8, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.output.weight": normal_array(
                    lseed + 9, [d, d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.output.bias": normal_array(
                    lseed + 10, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.norm_2.gamma": normal_array(
                    lseed + 11, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.norm_2.beta": normal_array(
                    lseed + 12, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.pwff.linear_inner.weight": normal_array(
                    lseed + 13, [d, h], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.pwff.linear_inner.bias": normal_array(
                    lseed + 14, [h], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.pwff.linear_outer.weight": normal_array(
                    lseed + 15, [h, d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.pwff.linear_outer.bias": normal_array(
                    lseed + 16, [d], 0.0, 1.0
                ),
            }
        )
    tensors["final_norm.gamma"] = normal_array(seed + 100, [EMBEDDING_DIM], 0.0, 1.0)
    tensors["final_norm.beta"] = normal_array(seed + 200, [EMBEDDING_DIM], 0.0, 1.0)
    tensors["embedding.weight"] = normal_array(
        seed + 250, [VOCAB_SIZE, EMBEDDING_DIM], 0.0, 1.0
    )
    tensors["predictor.weight"] = normal_array(
        seed + 300, [EMBEDDING_DIM, VOCAB_SIZE], 0.0, 1.0
    )
    tensors["predictor.bias"] = normal_array(seed + 400, [VOCAB_SIZE], 0.0, 1.0)
    return tensors
