import torch
import torch.nn as nn

from soma_models.v1.configs import LayerConfig, PositionWiseFeedForwardConfig
from soma_models.torch.v1.modules.pwff import PositionWiseFeedForward
from soma_models.torch.v1.modules.attention import MultiHeadAttention


class Layer(nn.Module):
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.embedding_dim, eps=1e-5)
        self.attention = MultiHeadAttention(
            num_heads=config.num_heads,
            num_features=config.embedding_dim,
            dropout_rate=config.dropout_rate,
            max_wavelength=config.max_wavelength,
            scale_factor=config.scale_factor,
        )
        self.norm_2 = nn.LayerNorm(config.embedding_dim, eps=1e-5)
        self.pwff = PositionWiseFeedForward(
            PositionWiseFeedForwardConfig(
                dropout_rate=config.dropout_rate,
                embedding_dim=config.embedding_dim,
                pwff_hidden_dim=config.pwff_hidden_dim,
            )
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, context: torch.Tensor, positions: torch.Tensor, attn_mask: torch.Tensor):
        x = context
        residual_path = self.norm_1(x)
        residual_path = self.attention(residual_path, positions=positions, mask=attn_mask)
        residual_path = self.dropout(residual_path)
        x = x + residual_path
        residual_path = self.norm_2(x)
        residual_path = self.pwff(residual_path)
        residual_path = self.dropout(residual_path)
        x = x + residual_path
        return x
