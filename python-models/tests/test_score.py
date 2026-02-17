"""Tests for the score() function — full on-chain evaluation pipeline."""

import numpy as np
import torch
from flax import nnx
from safetensors.numpy import save

from tests.helpers import small_config, build_model_weights
from soma_models.v1.configs import ModelConfig
from soma_models.torch.v1.model import Model as TorchModel
from soma_models.torch.v1.loss import score as torch_score
from soma_models.torch.serde import load_safetensor_into
from soma_models.flax.v1.model import Model as FlaxModel
from soma_models.flax.v1.loss import score as flax_score
from soma_models.flax.serde import Serde as FlaxSerde


def _load_torch_model(cfg, weights_bytes):
    model = TorchModel(ModelConfig(**cfg))
    load_safetensor_into(model, weights_bytes)
    model.eval()
    return model


def _load_flax_model(cfg, weights_bytes):
    model = FlaxModel(ModelConfig(**cfg), rngs=nnx.Rngs(0))
    FlaxSerde(model).deserialize(weights_bytes)
    model.eval()
    return model


def test_score_cross_framework_identical():
    """score() should produce the same result in torch and flax."""
    cfg = small_config()
    weights_bytes = save(build_model_weights(
        42, cfg["num_layers"], cfg["embedding_dim"],
        cfg["pwff_hidden_dim"], cfg["vocab_size"],
    ))
    data = bytes(range(256)) * 2  # 512 bytes

    torch_loss, torch_emb = torch_score(
        _load_torch_model(cfg, weights_bytes), data, seed=7,
        seq_len=64, batch_size=4,
    )
    flax_loss, flax_emb = flax_score(
        _load_flax_model(cfg, weights_bytes), data, seed=7,
        seq_len=64, batch_size=4,
    )

    np.testing.assert_allclose(torch_loss, flax_loss, atol=1e-4)
    np.testing.assert_allclose(torch_emb, flax_emb, atol=1e-5)


def test_score_single_batch():
    """score() with data that fits in a single batch."""
    cfg = small_config()
    weights_bytes = save(build_model_weights(
        99, cfg["num_layers"], cfg["embedding_dim"],
        cfg["pwff_hidden_dim"], cfg["vocab_size"],
    ))
    data = bytes([10, 20, 30, 40, 50, 60, 70, 80])

    loss, emb = torch_score(
        _load_torch_model(cfg, weights_bytes), data, seed=42,
        seq_len=8, batch_size=4,
    )
    assert isinstance(loss, float)
    assert len(emb) == cfg["embedding_dim"]


def test_score_multi_batch_running_mean():
    """score() averages across batches using running mean."""
    cfg = small_config()
    weights_bytes = save(build_model_weights(
        77, cfg["num_layers"], cfg["embedding_dim"],
        cfg["pwff_hidden_dim"], cfg["vocab_size"],
    ))
    # 3 chunks of seq_len=4, batch_size=2 → 2 batches (2+1)
    data = bytes(range(12))

    loss, emb = torch_score(
        _load_torch_model(cfg, weights_bytes), data, seed=5,
        seq_len=4, batch_size=2,
    )
    assert isinstance(loss, float)
    assert np.isfinite(loss)
    assert len(emb) == cfg["embedding_dim"]
    assert all(np.isfinite(e) for e in emb)


def test_score_empty_data_raises():
    """score() should raise ValueError on empty data."""
    cfg = small_config()
    weights_bytes = save(build_model_weights(
        42, cfg["num_layers"], cfg["embedding_dim"],
        cfg["pwff_hidden_dim"], cfg["vocab_size"],
    ))
    model = _load_torch_model(cfg, weights_bytes)

    try:
        torch_score(model, b"", seed=1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
