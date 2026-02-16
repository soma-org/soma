import torch
import torch.nn.functional as F
from arrgen import normal_array
from soma_models.config import (
    V1_PAD_TOKEN_ID,
    V1_SIG_REG_COEFFICIENT,
    V1_MAX_SEQ_LEN,
    V1_BATCH_SIZE,
)
from soma_models.v1.configs import SIGRegConfig
from soma_models.torch.v1.model import Model
from soma_models.torch.v1.sig_reg import SIGReg


def compute_loss(
    model: Model,
    sig_reg: SIGReg,
    sig_reg_config: SIGRegConfig,
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    attn_mask: torch.Tensor,
    targets: torch.Tensor,
    seed: int,
    sig_reg_coefficient: float = V1_SIG_REG_COEFFICIENT,
) -> tuple[torch.Tensor, torch.Tensor]:
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
            loss: Scalar tensor = cross_entropy + sig_reg_coefficient * sig_reg_loss
            embedding: Mean embedding, shape [embedding_dim]
    """
    representations = model(token_ids, positions, attn_mask)
    logits = model.predict(representations)

    # SIGReg: deterministic noise from arrgen, matching Rust runtime
    noise_data = normal_array(
        seed, [model.config.embedding_dim, sig_reg_config.slices], 0.0, 1.0
    )
    noise = torch.tensor(
        noise_data, dtype=representations.dtype, device=representations.device
    )
    sig_reg_loss = sig_reg(representations, noise)

    # Mean embedding: [batch, seq, embed] -> [embed]
    embedding = representations.mean(dim=1).mean(dim=0)

    # Cross entropy with PAD masking, matching Rust CrossEntropyLoss
    batch_size, seq, vocab = logits.shape
    logits_flat = logits.reshape(batch_size * seq, vocab)
    targets_flat = targets.reshape(batch_size * seq)
    ce_loss = F.cross_entropy(logits_flat, targets_flat.long(), ignore_index=V1_PAD_TOKEN_ID)

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
    from soma_models.v1.data import prepare_batches

    batches = prepare_batches(data, seq_len=seq_len, batch_size=batch_size)
    if not batches:
        raise ValueError("data must not be empty")

    sig_reg_config = SIGRegConfig()
    sig_reg = SIGReg(sig_reg_config)
    sig_reg.eval()

    device = next(model.parameters()).device

    loss_mean = None
    emb_mean = None
    count = 0

    with torch.no_grad():
        for batch in batches:
            token_ids = torch.tensor(batch["token_ids"], device=device)
            positions = torch.tensor(batch["positions"], device=device)
            attn_mask = torch.tensor(batch["attn_mask"], device=device)
            targets = torch.tensor(batch["targets"], device=device)

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

    return loss_mean.item(), emb_mean.tolist()
