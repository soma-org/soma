import torch
from arrgen import (
    normal_array,
    constant_array,
)
from safetensors.numpy import save
from soma_models.v1.torch.serde import load_safetensor_into
from soma_models.v1.torch.modules.pwff import (
    PositionWiseFeedForward,
    PositionWiseFeedForwardConfig,
)


def test_v1_pwff_ones():
    embedding_dim = 4
    hidden_dim = 2
    seed = 42
    module = PositionWiseFeedForward(
        PositionWiseFeedForwardConfig(
            dropout_rate=0.0,
            embedding_dim=embedding_dim,
            pwff_hidden_dim=hidden_dim,
        )
    )
    generated_tensors = {
        "linear_inner.weight": normal_array(
            seed + 1, [embedding_dim, hidden_dim], mean=0.0, std_dev=1.0
        ),
        "linear_inner.bias": normal_array(
            seed + 2, [hidden_dim], mean=0.0, std_dev=1.0
        ),
        "linear_outer.weight": normal_array(
            seed + 3, [hidden_dim, embedding_dim], mean=0.0, std_dev=1.0
        ),
        "linear_outer.bias": normal_array(
            seed + 4, [embedding_dim], mean=0.0, std_dev=1.0
        ),
    }
    serialized_tensors = save(generated_tensors)
    load_safetensor_into(module, serialized_tensors)
    module.eval()

    expected = torch.tensor([0.31442693, 0.70205802, -2.13397980, -1.71679294])
    input = torch.tensor(constant_array([embedding_dim], 1.0))
    output = module(input)

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"


def test_v1_pwff_normal_input():
    embedding_dim = 4
    hidden_dim = 2
    seed = 50
    module = PositionWiseFeedForward(
        PositionWiseFeedForwardConfig(
            dropout_rate=0.0,
            embedding_dim=embedding_dim,
            pwff_hidden_dim=hidden_dim,
        )
    )
    generated_tensors = {
        "linear_inner.weight": normal_array(
            seed + 1, [embedding_dim, hidden_dim], mean=0.0, std_dev=1.0
        ),
        "linear_inner.bias": normal_array(
            seed + 2, [hidden_dim], mean=0.0, std_dev=1.0
        ),
        "linear_outer.weight": normal_array(
            seed + 3, [hidden_dim, embedding_dim], mean=0.0, std_dev=1.0
        ),
        "linear_outer.bias": normal_array(
            seed + 4, [embedding_dim], mean=0.0, std_dev=1.0
        ),
    }
    serialized_tensors = save(generated_tensors)
    load_safetensor_into(module, serialized_tensors)
    module.eval()

    expected = torch.tensor([-0.42594010, -0.70958626, -0.26518542, -0.35035765])
    input = torch.tensor(normal_array(seed + 5, [embedding_dim], 0.0, 1.0))
    output = module(input)

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"


def test_v1_pwff_batched():
    embedding_dim = 4
    hidden_dim = 2
    seed = 60
    batch_size = 2
    seq_len = 3
    module = PositionWiseFeedForward(
        PositionWiseFeedForwardConfig(
            dropout_rate=0.0,
            embedding_dim=embedding_dim,
            pwff_hidden_dim=hidden_dim,
        )
    )
    generated_tensors = {
        "linear_inner.weight": normal_array(
            seed + 1, [embedding_dim, hidden_dim], mean=0.0, std_dev=1.0
        ),
        "linear_inner.bias": normal_array(
            seed + 2, [hidden_dim], mean=0.0, std_dev=1.0
        ),
        "linear_outer.weight": normal_array(
            seed + 3, [hidden_dim, embedding_dim], mean=0.0, std_dev=1.0
        ),
        "linear_outer.bias": normal_array(
            seed + 4, [embedding_dim], mean=0.0, std_dev=1.0
        ),
    }
    serialized_tensors = save(generated_tensors)
    load_safetensor_into(module, serialized_tensors)
    module.eval()

    expected = torch.tensor(
        [
            [
                [2.99256825, -2.32797265, -3.22312760, 1.21508217],
                [2.70691204, -2.19828272, -3.50960517, 0.83996159],
                [-0.04863004, 0.30158886, -1.44390535, 1.00410569],
            ],
            [
                [0.16999049, 0.10190730, -1.61299801, 0.98700720],
                [0.09025692, 0.22779667, -1.34613645, 1.15397000],
                [0.28953564, -0.03523720, -1.81356251, 0.89298093],
            ],
        ]
    )
    input = torch.tensor(
        normal_array(seed + 5, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    output = module(input)

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"
