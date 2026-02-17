import jax
import jax.numpy as jnp
from jax import Array

from soma_models.v1.configs import V1_PAD_TOKEN_ID
from soma_models.v1.flax.modules.model import Model
from soma_models.v1.flax.modules.sig_reg import SIGReg


def compute_loss(
    model: Model,
    sig_reg: SIGReg,
    token_ids: Array,
    targets: Array,
) -> tuple[Array, Array]:
    """Compute the training loss (cross-entropy + SIGReg regularization).

    Args:
        model: The V1 Model instance.
        sig_reg: The SIGReg module instance (generates noise internally via rngs).
        token_ids: Input token ids, shape [batch, seq].
        targets: Next-token targets, shape [batch, seq].

    Returns:
        (loss, embedding) where:
            loss: Scalar = cross_entropy + sig_reg_loss
            embedding: Mean embedding, shape [embedding_dim]
    """
    batch_size, seq_len = token_ids.shape
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))

    representations = model.encode(token_ids, positions)
    logits = model.predict(representations)
    sig_reg_loss = sig_reg(representations)

    embedding = jnp.mean(jnp.mean(representations, axis=1), axis=0)

    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    mask = targets_flat != V1_PAD_TOKEN_ID

    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    ce_per_token = -log_probs[jnp.arange(logits_flat.shape[0]), targets_flat]
    ce_loss = jnp.sum(ce_per_token * mask) / jnp.maximum(jnp.sum(mask), 1)

    return ce_loss + sig_reg_loss, embedding
