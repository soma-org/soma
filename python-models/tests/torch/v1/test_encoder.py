import torch
from arrgen import (
    normal_array,
)
from safetensors.numpy import save
from soma_models.torch.serde import load_safetensor_into
from soma_models.torch.v1.modules.encoder import Encoder, EncoderConfig


def test_v1_attention():
    seed = 42
    batch_size = 1
    seq_len = 4
    num_heads = 2
    head_dim = 2
    num_layers = 2
    embedding_dim = head_dim * num_heads
    hidden_dim = embedding_dim * 2
    module = Encoder(
        EncoderConfig(
            dropout_rate=0.0,
            embedding_dim=embedding_dim,
            pwff_hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
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
    load_safetensor_into(module, serialized_tensors)
    module.eval()
    inputs = torch.tensor(
        normal_array(seed + 100, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    positions = torch.arange(0, seq_len).reshape(1, -1).repeat(batch_size, 1)
    mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).tril()
    outputs = module(inputs, positions=positions, attn_mask=mask)
    print(outputs)
    expected = torch.tensor(
        [
            [
                [8.33097267, 3.23748636, 12.93889809, -4.79724693],
                [5.95732164, 2.65965629, 11.97813892, -3.51363301],
                [8.32810211, 4.93732452, 12.48622322, -4.29308748],
                [3.45333171, 3.12663937, 10.79409981, -2.80449820],
            ]
        ],
    )
    assert torch.allclose(outputs, expected), f"Arrays are not close enough! {outputs} vs {expected}"
