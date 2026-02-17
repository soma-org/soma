import torch
import torch.nn.functional as F

from soma_models.v1.configs import V1_PAD_TOKEN_ID
from soma_models.v1.torch.modules.model import Model
from soma_models.v1.torch.modules.sig_reg import SIGReg


def compute_loss(
    model: Model,
    sig_reg: SIGReg,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the training loss (cross-entropy + SIGReg regularization).

    Args:
        model: The V1 Model instance.
        sig_reg: The SIGReg module instance (generates noise internally via global RNG).
        token_ids: Input token ids, shape [batch, seq].
        targets: Next-token targets, shape [batch, seq].

    Returns:
        (loss, embedding) where:
            loss: Scalar tensor = cross_entropy + sig_reg_loss
            embedding: Mean embedding, shape [embedding_dim]
    """
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)

    representations = model.encode(token_ids, positions)
    logits = model.predict(representations)
    sig_reg_loss = sig_reg(representations)

    embedding = representations.mean(dim=1).mean(dim=0)

    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    ce_loss = F.cross_entropy(logits_flat, targets_flat.long(), ignore_index=V1_PAD_TOKEN_ID)

    return ce_loss + sig_reg_loss, embedding
