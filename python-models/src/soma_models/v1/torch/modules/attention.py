import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def apply_rope(
    inputs: torch.Tensor,  # [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM]
    positions: torch.Tensor,  # [BATCH_SIZE, SEQ_LEN]
    head_dim: int,
    max_wavelength: int = 10_000,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    fraction = 2 * torch.arange(0, head_dim // 2, dtype=inputs.dtype, device=inputs.device) / head_dim
    timescale = max_wavelength ** fraction
    positions = positions.unsqueeze(-1).to(inputs.dtype)  # [batch, seq, 1]
    timescale = timescale.unsqueeze(0).unsqueeze(0)  # [1, 1, head_dim//2]

    sinusoid_inp = positions / timescale
    sinusoid_inp = sinusoid_inp.unsqueeze(-2)  # [batch, seq, 1, head_dim//2]
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
    sinusoid_inp = sinusoid_inp / scale_factor

    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)

    first_half, second_half = inputs.chunk(2, dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = torch.cat([first_part, second_part], dim=-1)
    return out.to(inputs.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_features: int,
        *,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        max_wavelength: float = 10_000.0,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_features = num_features
        self.dropout_rate = dropout_rate
        self.max_wavelength = max_wavelength
        self.scale_factor = scale_factor

        if num_features % num_heads != 0:
            raise ValueError(
                f"Memory dimension ({num_features}) must be divisible by "
                f"'num_heads' heads ({num_heads})."
            )
        self.head_dim = num_features // num_heads

        self.query = nn.Linear(num_features, num_features, bias=use_bias)
        self.key = nn.Linear(num_features, num_features, bias=use_bias)
        self.value = nn.Linear(num_features, num_features, bias=use_bias)
        self.output = nn.Linear(num_features, num_features, bias=use_bias)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
    ):
        batch_size, seq_len, _ = inputs.shape

        # Project to Q, K, V and reshape to [batch, seq, heads, head_dim]
        query = self.query(inputs).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key(inputs).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value(inputs).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        if positions is not None:
            query = apply_rope(query, positions, head_dim=self.head_dim, max_wavelength=self.max_wavelength, scale_factor=self.scale_factor)
            key = apply_rope(key, positions, head_dim=self.head_dim, max_wavelength=self.max_wavelength, scale_factor=self.scale_factor)

        # Transpose to [batch, heads, seq, head_dim] for attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Scale
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / scale

        # Generate causal mask: [batch, 1, seq, seq] with True=attend, False=mask
        mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=inputs.device).tril()
        big_neg = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(mask, attn_weights, big_neg)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.training and self.dropout_rate > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate)

        # [batch, heads, seq, head_dim]
        attn_output = torch.matmul(attn_weights, value)

        # Transpose back to [batch, seq, heads, head_dim] and reshape
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_features)

        out = self.output(attn_output)
        return out
