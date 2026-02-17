import jax.numpy as jnp
import optax
from jax import Array
from arrgen import normal_array
from soma_models.config import (
    V1_PAD_TOKEN_ID,
    V1_SIG_REG_COEFFICIENT,
    V1_MAX_SEQ_LEN,
    V1_BATCH_SIZE,
)
from soma_models.v1.configs import SIGRegConfig
from soma_models.flax.v1.model import Model
from soma_models.flax.v1.sig_reg import SIGReg


def compute_loss(
    model: Model,
    sig_reg: SIGReg,
    sig_reg_config: SIGRegConfig,
    token_ids: Array,
    positions: Array,
    attn_mask: Array,
    targets: Array,
    seed: int,
    sig_reg_coefficient: float = V1_SIG_REG_COEFFICIENT,
) -> tuple[Array, Array]:
    """Compute the scoring loss that mirrors the Rust ModelRunner::eval.

    This is the exact loss function used on-chain to evaluate models.
    It is differentiable — gradients flow through both components.

    Args:
        model: The V1 Model instance.
        sig_reg: The SIGReg module instance.
        sig_reg_config: The SIGReg config (needed for slices dimension).
        token_ids: Input token ids, shape [batch, seq].
        positions: Position ids, shape [batch, seq].
        attn_mask: Causal attention mask, shape [batch, 1, seq, seq].
        targets: Next-token targets, shape [batch, seq].
        seed: Deterministic seed for SIGReg noise generation.
        sig_reg_coefficient: Weight for the SIGReg term.

    Returns:
        (loss, embedding) where:
            loss: Scalar = cross_entropy + sig_reg_coefficient * sig_reg_loss
            embedding: Mean embedding, shape [embedding_dim]
    """
    representations = model(token_ids, positions, attn_mask)
    logits = model.predict(representations)

    # SIGReg: deterministic noise from arrgen, matching Rust runtime
    noise_data = normal_array(
        seed, [model.config.embedding_dim, sig_reg_config.slices], 0.0, 1.0
    )
    noise = jnp.array(noise_data, dtype=representations.dtype)
    sig_reg_loss = sig_reg(representations, noise)

    # Mean embedding: [batch, seq, embed] -> [embed]
    embedding = jnp.mean(jnp.mean(representations, axis=1), axis=0)

    # Cross entropy with PAD masking, matching Rust CrossEntropyLoss
    batch_size, seq, vocab = logits.shape
    logits_flat = logits.reshape(batch_size * seq, vocab)
    targets_flat = targets.reshape(batch_size * seq)

    # Mask out PAD tokens
    mask = targets_flat != V1_PAD_TOKEN_ID
    ce_per_token = optax.softmax_cross_entropy_with_integer_labels(
        logits_flat, targets_flat, where=mask
    )
    ce_loss = jnp.sum(ce_per_token) / jnp.maximum(jnp.sum(mask), 1)

    loss = ce_loss + sig_reg_loss * sig_reg_coefficient

    return loss, embedding


def score(
    model: Model,
    data: bytes,
    seed: int,
    seq_len: int = V1_MAX_SEQ_LEN,
    batch_size: int = V1_BATCH_SIZE,
) -> tuple[float, list[float]]:
    """Score a model on raw data, replicating the Rust ModelRunner::eval exactly.

    This runs the full on-chain evaluation pipeline: chunk bytes into sequences,
    construct position IDs and causal masks, compute loss per batch, and average
    across batches using Welford's running mean (matching the Rust RunningMean).

    Args:
        model: The V1 Model instance (should be in eval mode with dropout=0).
        data: Raw bytes to evaluate.
        seed: Deterministic seed for SIGReg noise generation.
        seq_len: Sequence length per chunk (default: V1_MAX_SEQ_LEN).
        batch_size: Sequences per batch (default: V1_BATCH_SIZE).

    Returns:
        (loss, embedding) where:
            loss: float — the mean loss across all batches.
            embedding: list[float] — the mean embedding vector.

    Raises:
        ValueError: If data is empty.
    """
    from flax import nnx
    from soma_models.v1.data import prepare_batches

    batches = prepare_batches(data, seq_len=seq_len, batch_size=batch_size)
    if not batches:
        raise ValueError("data must not be empty")

    sig_reg_config = SIGRegConfig()
    sig_reg = SIGReg(sig_reg_config, rngs=nnx.Rngs(0))

    loss_mean = None
    emb_mean = None
    count = 0

    for batch in batches:
        token_ids = jnp.array(batch["token_ids"])
        positions = jnp.array(batch["positions"])
        attn_mask = jnp.array(batch["attn_mask"])
        targets = jnp.array(batch["targets"])

        batch_loss, batch_emb = compute_loss(
            model, sig_reg, sig_reg_config,
            token_ids, positions, attn_mask, targets,
            seed=seed,
        )

        count += 1
        if loss_mean is None:
            loss_mean = batch_loss
            emb_mean = batch_emb
        else:
            loss_mean = loss_mean + (batch_loss - loss_mean) / count
            emb_mean = emb_mean + (batch_emb - emb_mean) / count

    return float(loss_mean), emb_mean.tolist()
