"""Verify that loss computation produces identical results across Torch and Flax."""

import numpy as np
import torch
from flax import nnx
from safetensors.numpy import save

from tests.helpers import small_config, build_model_weights
from soma_models.config import V1_PAD_TOKEN_ID, V1_SIG_REG_COEFFICIENT
from soma_models.v1.configs import ModelConfig, SIGRegConfig
from soma_models.torch.v1.model import Model as TorchModel
from soma_models.torch.v1.sig_reg import SIGReg as TorchSIGReg
from soma_models.torch.v1.loss import compute_loss as torch_compute_loss
from soma_models.torch.serde import load_safetensor_into
from soma_models.flax.v1.model import Model as FlaxModel
from soma_models.flax.v1.sig_reg import SIGReg as FlaxSIGReg
from soma_models.flax.v1.loss import compute_loss as flax_compute_loss
from soma_models.flax.serde import Serde as FlaxSerde


def test_cross_framework_loss_identical():
    """Both frameworks should produce the same loss and embedding given identical weights."""
    cfg = small_config()
    weight_seed = 42
    eval_seed = 7
    seq_len = 8
    batch_size = 2
    sig_reg_slices = 32
    sig_reg_points = 5

    weights = build_model_weights(
        weight_seed, cfg["num_layers"], cfg["embedding_dim"],
        cfg["pwff_hidden_dim"], cfg["vocab_size"],
    )
    weights_bytes = save(weights)

    # Deterministic input: byte tokens with next-token targets
    token_ids_np = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [10, 20, 30, 40, 50, 60, 70, 80],
    ], dtype=np.int32)
    targets_np = np.array([
        [1, 2, 3, 4, 5, 6, 7, V1_PAD_TOKEN_ID],
        [20, 30, 40, 50, 60, 70, 80, V1_PAD_TOKEN_ID],
    ], dtype=np.int32)

    # --- Torch ---
    torch_model = TorchModel(ModelConfig(**cfg))
    load_safetensor_into(torch_model, weights_bytes)
    torch_model.eval()

    torch_sig_reg_config = SIGRegConfig(slices=sig_reg_slices, points=sig_reg_points)
    torch_sig_reg = TorchSIGReg(torch_sig_reg_config)
    torch_sig_reg.eval()

    torch_tokens = torch.tensor(token_ids_np)
    torch_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    torch_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).tril()
    torch_targets = torch.tensor(targets_np)

    with torch.no_grad():
        torch_loss, torch_embedding = torch_compute_loss(
            torch_model, torch_sig_reg, torch_sig_reg_config,
            torch_tokens, torch_positions, torch_mask, torch_targets,
            seed=eval_seed, sig_reg_coefficient=V1_SIG_REG_COEFFICIENT,
        )

    # --- Flax ---
    import jax.numpy as jnp

    flax_model = FlaxModel(ModelConfig(**cfg), rngs=nnx.Rngs(0))
    FlaxSerde(flax_model).deserialize(weights_bytes)
    flax_model.eval()

    flax_sig_reg_config = SIGRegConfig(slices=sig_reg_slices, points=sig_reg_points)
    flax_sig_reg = FlaxSIGReg(flax_sig_reg_config, rngs=nnx.Rngs(0))

    flax_tokens = jnp.array(token_ids_np)
    flax_positions = jnp.tile(jnp.arange(seq_len).reshape(1, -1), (batch_size, 1))
    flax_mask = nnx.make_causal_mask(flax_positions, dtype=bool)
    flax_targets = jnp.array(targets_np)

    flax_loss, flax_embedding = flax_compute_loss(
        flax_model, flax_sig_reg, flax_sig_reg_config,
        flax_tokens, flax_positions, flax_mask, flax_targets,
        seed=eval_seed, sig_reg_coefficient=V1_SIG_REG_COEFFICIENT,
    )

    # --- Compare ---
    torch_loss_val = torch_loss.item()
    flax_loss_val = float(flax_loss)
    np.testing.assert_allclose(
        torch_loss_val, flax_loss_val, atol=1e-4,
        err_msg=f"Loss mismatch: torch={torch_loss_val}, flax={flax_loss_val}",
    )

    torch_emb = torch_embedding.numpy()
    flax_emb = np.array(flax_embedding)
    np.testing.assert_allclose(
        torch_emb, flax_emb, atol=1e-5,
        err_msg="Embedding mismatch between torch and flax",
    )


def test_cross_framework_loss_with_padding():
    """Loss should handle sequences with varying amounts of padding identically."""
    cfg = small_config()
    weight_seed = 99
    eval_seed = 13
    seq_len = 6
    batch_size = 2
    sig_reg_slices = 16
    sig_reg_points = 5

    weights = build_model_weights(
        weight_seed, cfg["num_layers"], cfg["embedding_dim"],
        cfg["pwff_hidden_dim"], cfg["vocab_size"],
    )
    weights_bytes = save(weights)

    # Second sequence has heavy padding (only 3 real tokens)
    token_ids_np = np.array([
        [0, 1, 2, 3, 4, 5],
        [10, 20, 30, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID],
    ], dtype=np.int32)
    targets_np = np.array([
        [1, 2, 3, 4, 5, V1_PAD_TOKEN_ID],
        [20, 30, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID],
    ], dtype=np.int32)

    # --- Torch ---
    torch_model = TorchModel(ModelConfig(**cfg))
    load_safetensor_into(torch_model, weights_bytes)
    torch_model.eval()

    torch_sig_reg_config = SIGRegConfig(slices=sig_reg_slices, points=sig_reg_points)
    torch_sig_reg = TorchSIGReg(torch_sig_reg_config)
    torch_sig_reg.eval()

    torch_tokens = torch.tensor(token_ids_np)
    torch_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    torch_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).tril()
    torch_targets = torch.tensor(targets_np)

    with torch.no_grad():
        torch_loss, torch_embedding = torch_compute_loss(
            torch_model, torch_sig_reg, torch_sig_reg_config,
            torch_tokens, torch_positions, torch_mask, torch_targets,
            seed=eval_seed,
        )

    # --- Flax ---
    import jax.numpy as jnp

    flax_model = FlaxModel(ModelConfig(**cfg), rngs=nnx.Rngs(0))
    FlaxSerde(flax_model).deserialize(weights_bytes)
    flax_model.eval()

    flax_sig_reg_config = SIGRegConfig(slices=sig_reg_slices, points=sig_reg_points)
    flax_sig_reg = FlaxSIGReg(flax_sig_reg_config, rngs=nnx.Rngs(0))

    flax_tokens = jnp.array(token_ids_np)
    flax_positions = jnp.tile(jnp.arange(seq_len).reshape(1, -1), (batch_size, 1))
    flax_mask = nnx.make_causal_mask(flax_positions, dtype=bool)
    flax_targets = jnp.array(targets_np)

    flax_loss, flax_embedding = flax_compute_loss(
        flax_model, flax_sig_reg, flax_sig_reg_config,
        flax_tokens, flax_positions, flax_mask, flax_targets,
        seed=eval_seed,
    )

    # --- Compare ---
    np.testing.assert_allclose(
        torch_loss.item(), float(flax_loss), atol=1e-4,
        err_msg="Loss mismatch with padding",
    )
    np.testing.assert_allclose(
        torch_embedding.numpy(), np.array(flax_embedding), atol=1e-5,
        err_msg="Embedding mismatch with padding",
    )
