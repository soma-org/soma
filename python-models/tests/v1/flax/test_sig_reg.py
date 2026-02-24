from jax import numpy as jnp
from arrgen import normal_array, uniform_array
from flax import nnx
from soma_models.v1.flax.modules.sig_reg import (
    SIGReg,
    SIGRegConfig,
)


def test_v1_sig_reg_normal():
    seed = 42
    batch_size = 10
    seq_len = 10
    embedding_dim = 1024

    sig_reg_config = SIGRegConfig(coefficient=1.0)
    sig_reg = SIGReg(sig_reg_config, rngs=nnx.Rngs(0))
    input = jnp.array(
        normal_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = jnp.array(
        normal_array(seed + 2, [embedding_dim, sig_reg_config.slices], 0.0, 1.0)
    )

    output = sig_reg.compute(input, noise)
    expected = jnp.array(
        [1.28620601],
    )

    assert jnp.allclose(output, expected), "Arrays are not close enough!"


def test_v1_sig_reg_small_dim():
    seed = 99
    batch_size = 2
    seq_len = 3
    embedding_dim = 8

    sig_reg_config = SIGRegConfig(slices=4, points=5, coefficient=1.0)
    sig_reg = SIGReg(sig_reg_config, rngs=nnx.Rngs(0))
    input = jnp.array(
        normal_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = jnp.array(
        normal_array(seed + 2, [embedding_dim, sig_reg_config.slices], 0.0, 1.0)
    )

    output = sig_reg.compute(input, noise)
    expected = jnp.array(
        [0.88936800],
    )

    assert jnp.allclose(output, expected), "Arrays are not close enough!"


def test_v1_sig_reg_single_batch():
    seed = 110
    batch_size = 1
    seq_len = 1
    embedding_dim = 16

    sig_reg_config = SIGRegConfig(coefficient=1.0)
    sig_reg = SIGReg(sig_reg_config, rngs=nnx.Rngs(0))
    input = jnp.array(
        normal_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = jnp.array(
        normal_array(seed + 2, [embedding_dim, sig_reg_config.slices], 0.0, 1.0)
    )

    output = sig_reg.compute(input, noise)
    expected = jnp.array(
        [3.18785000],
    )

    assert jnp.allclose(output, expected), "Arrays are not close enough!"


def test_v1_sig_reg_uniform():
    seed = 42
    batch_size = 10
    seq_len = 10
    embedding_dim = 1024

    sig_reg_config = SIGRegConfig(coefficient=1.0)
    sig_reg = SIGReg(sig_reg_config, rngs=nnx.Rngs(0))
    input = jnp.array(
        uniform_array(seed + 1, [batch_size, seq_len, embedding_dim], 0.0, 1.0)
    )
    noise = jnp.array(
        normal_array(seed + 2, [embedding_dim, sig_reg_config.slices], 0.0, 1.0)
    )

    output = sig_reg.compute(input, noise)
    expected = jnp.array(
        [29.08621025],
    )

    assert jnp.allclose(output, expected), "Arrays are not close enough!"
