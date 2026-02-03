from jax import numpy as jnp
from arrgen import normal_array, uniform_array
from flax import nnx
from soma_models.flax.v1.sig_reg import (
    SIGReg,
    SIGRegConfig,
)


def test_v1_sig_reg_normal():
    seed = 42
    batch_size = 10
    seq_len = 10
    embedding_dim = 1024

    sig_reg_config = SIGRegConfig()
    sig_reg = SIGReg(sig_reg_config, rngs=nnx.Rngs(0))
    input = jnp.array(
        normal_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = jnp.array(
        normal_array(seed + 2, [embedding_dim, sig_reg_config.slices], 0.0, 1.0)
    )

    output = sig_reg(input, noise)
    expected = jnp.array(
        [1.33955204],
    )

    assert jnp.allclose(output, expected), "Arrays are not close enough!"


def test_v1_sig_reg_uniform():
    seed = 42
    batch_size = 10
    seq_len = 10
    embedding_dim = 1024

    sig_reg_config = SIGRegConfig()
    sig_reg = SIGReg(sig_reg_config, rngs=nnx.Rngs(0))
    input = jnp.array(
        uniform_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = jnp.array(
        normal_array(seed + 2, [embedding_dim, sig_reg_config.slices], 0.0, 1.0)
    )

    output = sig_reg(input, noise)
    expected = jnp.array(
        [121.57125092],
    )

    assert jnp.allclose(output, expected), "Arrays are not close enough!"
