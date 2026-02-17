import torch
from arrgen import normal_array, uniform_array
from soma_models.v1.torch.modules.sig_reg import (
    SIGReg,
    SIGRegConfig,
)


def test_v1_sig_reg_normal():
    seed = 42
    batch_size = 10
    seq_len = 10
    embedding_dim = 1024

    sig_reg_config = SIGRegConfig(coefficient=1.0)
    sig_reg = SIGReg(sig_reg_config)
    sig_reg.eval()
    input = torch.tensor(
        normal_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = torch.tensor(
        normal_array(seed + 2, [embedding_dim, sig_reg_config.slices], 0.0, 1.0)
    )

    output = sig_reg.compute(input, noise)
    expected = torch.tensor(
        [1.33955204],
    )

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"


def test_v1_sig_reg_uniform():
    seed = 42
    batch_size = 10
    seq_len = 10
    embedding_dim = 1024

    sig_reg_config = SIGRegConfig(coefficient=1.0)
    sig_reg = SIGReg(sig_reg_config)
    sig_reg.eval()
    input = torch.tensor(
        uniform_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = torch.tensor(
        normal_array(seed + 2, [embedding_dim, sig_reg_config.slices], 0.0, 1.0)
    )

    output = sig_reg.compute(input, noise)
    expected = torch.tensor(
        [121.57125092],
    )

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"


def test_v1_sig_reg_small_dim():
    seed = 99
    batch_size = 2
    seq_len = 3
    embedding_dim = 8
    slices = 4
    points = 5

    sig_reg_config = SIGRegConfig(slices=slices, points=points, coefficient=1.0)
    sig_reg = SIGReg(sig_reg_config)
    sig_reg.eval()
    input = torch.tensor(
        normal_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = torch.tensor(
        normal_array(seed + 2, [embedding_dim, slices], 0.0, 1.0)
    )

    output = sig_reg.compute(input, noise)
    expected = torch.tensor(
        [0.88936800],
    )

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"


def test_v1_sig_reg_single_batch():
    seed = 110
    batch_size = 1
    seq_len = 1
    embedding_dim = 16

    sig_reg_config = SIGRegConfig(coefficient=1.0)
    sig_reg = SIGReg(sig_reg_config)
    sig_reg.eval()
    input = torch.tensor(
        normal_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = torch.tensor(
        normal_array(seed + 2, [embedding_dim, sig_reg_config.slices], 0.0, 1.0)
    )

    output = sig_reg.compute(input, noise)
    expected = torch.tensor(
        [17.77822685],
    )

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"
