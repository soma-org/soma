"""Verify that loss computation produces identical results across Torch and Flax."""

import numpy as np
import torch
from flax import nnx
from safetensors.numpy import save
from arrgen import normal_array

from tests.v1.helpers import small_config, build_model_weights
from soma_models.v1.configs import V1_PAD_TOKEN_ID, ModelConfig, SIGRegConfig
from soma_models.v1.torch.modules.model import Model as TorchModel
from soma_models.v1.torch.modules.sig_reg import SIGReg as TorchSIGReg
from soma_models.v1.torch.loss import compute_loss as torch_compute_loss
from soma_models.v1.torch.serde import load_safetensor_into
from soma_models.v1.flax.modules.model import Model as FlaxModel
from soma_models.v1.flax.modules.sig_reg import SIGReg as FlaxSIGReg
from soma_models.v1.flax.loss import compute_loss as flax_compute_loss
from soma_models.v1.flax.serde import Serde as FlaxSerde


def _build_shared_fixtures(cfg, weight_seed, token_ids_np, targets_np):
    """Build torch and flax models with identical weights and inputs."""
    import jax.numpy as jnp

    weights = build_model_weights(
        weight_seed, cfg["num_layers"], cfg["embedding_dim"],
        cfg["pwff_hidden_dim"], cfg["vocab_size"],
    )
    weights_bytes = save(weights)

    torch_model = TorchModel(ModelConfig(**cfg))
    load_safetensor_into(torch_model, weights_bytes)
    torch_model.eval()

    flax_model = FlaxModel(ModelConfig(**cfg), rngs=nnx.Rngs(0))
    FlaxSerde(flax_model).deserialize(weights_bytes)
    flax_model.eval()

    return (
        torch_model, flax_model,
        torch.tensor(token_ids_np), jnp.array(token_ids_np),
        torch.tensor(targets_np), jnp.array(targets_np),
    )


def test_cross_framework_loss_identical():
    """Cross-entropy and embedding must match across frameworks given identical weights."""
    cfg = small_config()
    token_ids_np = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [10, 20, 30, 40, 50, 60, 70, 80],
    ], dtype=np.int32)
    targets_np = np.array([
        [1, 2, 3, 4, 5, 6, 7, V1_PAD_TOKEN_ID],
        [20, 30, 40, 50, 60, 70, 80, V1_PAD_TOKEN_ID],
    ], dtype=np.int32)

    torch_model, flax_model, torch_tokens, flax_tokens, torch_targets, flax_targets = \
        _build_shared_fixtures(cfg, 42, token_ids_np, targets_np)

    # Use coefficient=0 to isolate cross-entropy (eliminates noise-dependent SIGReg)
    torch_sig_reg = TorchSIGReg(SIGRegConfig(slices=32, points=5, coefficient=0.0))
    torch_sig_reg.eval()
    flax_sig_reg = FlaxSIGReg(SIGRegConfig(slices=32, points=5, coefficient=0.0), rngs=nnx.Rngs(0))

    with torch.no_grad():
        torch_loss, torch_embedding = torch_compute_loss(
            torch_model, torch_sig_reg, torch_tokens, torch_targets,
        )

    flax_loss, flax_embedding = flax_compute_loss(
        flax_model, flax_sig_reg, flax_tokens, flax_targets,
    )

    np.testing.assert_allclose(
        torch_loss.item(), float(flax_loss), atol=1e-4,
        err_msg=f"CE loss mismatch: torch={torch_loss.item()}, flax={float(flax_loss)}",
    )
    np.testing.assert_allclose(
        torch_embedding.numpy(), np.array(flax_embedding), atol=1e-5,
        err_msg="Embedding mismatch between torch and flax",
    )


def test_cross_framework_loss_with_padding():
    """Loss and embedding must match with varying padding amounts."""
    cfg = small_config()
    token_ids_np = np.array([
        [0, 1, 2, 3, 4, 5],
        [10, 20, 30, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID],
    ], dtype=np.int32)
    targets_np = np.array([
        [1, 2, 3, 4, 5, V1_PAD_TOKEN_ID],
        [20, 30, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID, V1_PAD_TOKEN_ID],
    ], dtype=np.int32)

    torch_model, flax_model, torch_tokens, flax_tokens, torch_targets, flax_targets = \
        _build_shared_fixtures(cfg, 99, token_ids_np, targets_np)

    torch_sig_reg = TorchSIGReg(SIGRegConfig(slices=16, points=5, coefficient=0.0))
    torch_sig_reg.eval()
    flax_sig_reg = FlaxSIGReg(SIGRegConfig(slices=16, points=5, coefficient=0.0), rngs=nnx.Rngs(0))

    with torch.no_grad():
        torch_loss, torch_embedding = torch_compute_loss(
            torch_model, torch_sig_reg, torch_tokens, torch_targets,
        )

    flax_loss, flax_embedding = flax_compute_loss(
        flax_model, flax_sig_reg, flax_tokens, flax_targets,
    )

    np.testing.assert_allclose(
        torch_loss.item(), float(flax_loss), atol=1e-4,
        err_msg="CE loss mismatch with padding",
    )
    np.testing.assert_allclose(
        torch_embedding.numpy(), np.array(flax_embedding), atol=1e-5,
        err_msg="Embedding mismatch with padding",
    )


def test_sig_reg_math_cross_framework():
    """SIGReg math (via compute) must be identical across frameworks given the same arrgen noise."""
    import jax.numpy as jnp

    embedding_dim = 16
    slices = 8
    sig_reg_config = SIGRegConfig(slices=slices, points=5, coefficient=1.0)
    seed = 42

    noise_data = normal_array(seed, [embedding_dim, slices], 0.0, 1.0)

    rng = np.random.RandomState(99)
    x_np = rng.randn(2, 4, embedding_dim).astype(np.float32)

    torch_sig_reg = TorchSIGReg(sig_reg_config)
    torch_sig_reg.eval()
    with torch.no_grad():
        torch_result = torch_sig_reg.compute(
            torch.tensor(x_np),
            torch.tensor(np.array(noise_data, dtype=np.float32)),
        )

    flax_sig_reg = FlaxSIGReg(sig_reg_config, rngs=nnx.Rngs(0))
    flax_result = flax_sig_reg.compute(
        jnp.array(x_np),
        jnp.array(np.array(noise_data, dtype=np.float32)),
    )

    np.testing.assert_allclose(
        torch_result.item(), float(flax_result), atol=1e-5,
        err_msg="SIGReg math mismatch between torch and flax with identical arrgen noise",
    )
