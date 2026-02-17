import torch
from safetensors.numpy import save
from tests.helpers import build_model_weights
from soma_models.torch.serde import load_safetensor_into
from soma_models.torch.v1.model import Model
from soma_models.v1.configs import ModelConfig


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
    generated_tensors = build_model_weights(
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
    generated_tensors = build_model_weights(
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
