import torch
from arrgen import normal_array, uniform_array
from soma_models.torch.v1.sig_reg import (
    SIGReg,
    SIGRegConfig,
)


def test_v1_sig_reg_normal():
    seed = 42
    batch_size = 10
    seq_len = 10
    embedding_dim = 1024

    sig_reg_config = SIGRegConfig()
    sig_reg = SIGReg(sig_reg_config)
    sig_reg.eval()
    input = torch.tensor(
        normal_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = torch.tensor(
        normal_array(seed + 2, [embedding_dim, sig_reg_config.slices], 0.0, 1.0)
    )

    output = sig_reg(input, noise)
    expected = torch.tensor(
        [1.33955204],
    )

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"


def test_v1_sig_reg_uniform():
    seed = 42
    batch_size = 10
    seq_len = 10
    embedding_dim = 1024

    sig_reg_config = SIGRegConfig()
    sig_reg = SIGReg(sig_reg_config)
    sig_reg.eval()
    input = torch.tensor(
        uniform_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = torch.tensor(
        normal_array(seed + 2, [embedding_dim, sig_reg_config.slices], 0.0, 1.0)
    )

    output = sig_reg(input, noise)
    expected = torch.tensor(
        [121.57125092],
    )

    assert torch.allclose(output, expected), f"Arrays are not close enough! {output} vs {expected}"
