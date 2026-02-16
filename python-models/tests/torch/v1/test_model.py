import torch
from arrgen import normal_array
from safetensors.numpy import save
from soma_models.torch.serde import load_safetensor_into
from soma_models.torch.v1.model import Model, ModelConfig


def _build_model_weights(seed, num_layers, embedding_dim, hidden_dim, vocab_size):
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


def test_v1_model():
    seed = 42
    batch_size = 1
    vocab_size = 4
    seq_len = 4
    num_heads = 2
    head_dim = 2
    num_layers = 2
    embedding_dim = num_heads * head_dim
    hidden_dim = embedding_dim * 2

    module = Model(
        ModelConfig(
            dropout_rate=0.0,
            embedding_dim=embedding_dim,
            pwff_hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
        )
    )
    generated_tensors = _build_model_weights(
        seed, num_layers, embedding_dim, hidden_dim, vocab_size
    )
    serialized_tensors = save(generated_tensors)
    load_safetensor_into(module, serialized_tensors)
    module.eval()

    tokens = torch.tensor([[0, 1, 2, 3]])
    positions = torch.arange(0, seq_len).reshape(1, -1).repeat(batch_size, 1)
    mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).tril()
    outputs = module(tokens, positions=positions, attn_mask=mask)
    expected = torch.tensor(
        [[
            [2.09210730, 0.69636524, 1.51327145, 2.31296515],
            [1.90634847, 0.86709338, 1.53078938, 2.40285897],
            [1.77797925, 0.66600311, 1.65812206, 2.19417143],
            [1.96223700, 0.77187395, 1.54670000, 2.34615612],
        ]],
    )
    assert torch.allclose(outputs, expected), f"Arrays are not close enough! {outputs} vs {expected}"


def test_v1_predict():
    seed = 42
    batch_size = 1
    vocab_size = 4
    seq_len = 4
    num_heads = 2
    head_dim = 2
    num_layers = 2
    embedding_dim = num_heads * head_dim
    hidden_dim = embedding_dim * 2

    module = Model(
        ModelConfig(
            dropout_rate=0.0,
            embedding_dim=embedding_dim,
            pwff_hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
        )
    )
    generated_tensors = _build_model_weights(
        seed, num_layers, embedding_dim, hidden_dim, vocab_size
    )
    serialized_tensors = save(generated_tensors)
    load_safetensor_into(module, serialized_tensors)
    module.eval()

    tokens = torch.tensor([[0, 1, 2, 3]])
    positions = torch.arange(0, seq_len).reshape(1, -1).repeat(batch_size, 1)
    mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).tril()
    embeddings = module(tokens, positions=positions, attn_mask=mask)
    outputs = module.predict(embeddings)
    expected = torch.tensor(
        [[
            [1.89312398, -3.37666154, -5.06565857, -9.11204624],
            [1.40491748, -3.30115676, -5.32568741, -8.60974312],
            [1.71110404, -2.86263919, -4.95488262, -8.56918144],
            [1.63806129, -3.27318168, -5.17767763, -8.81518459],
        ]],
    )
    assert torch.allclose(outputs, expected), f"Arrays are not close enough! {outputs} vs {expected}"
