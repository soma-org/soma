import torch
import torch.nn as nn
from dataclasses import dataclass
from soma_models.config import V1_NUM_LAYERS, V1_EMBEDDING_DIM, V1_PWFF_HIDDEN_DIM, V1_NUM_HEADS, V1_MAX_WAVELENGTH, V1_SCALE_FACTOR
from soma_models.torch.v1.modules.layer import Layer, LayerConfig


@dataclass
class EncoderConfig:
    dropout_rate: float
    embedding_dim: int = V1_EMBEDDING_DIM
    pwff_hidden_dim: int = V1_PWFF_HIDDEN_DIM
    num_layers: int = V1_NUM_LAYERS
    num_heads: int = V1_NUM_HEADS
    max_wavelength: float = V1_MAX_WAVELENGTH
    scale_factor: float = V1_SCALE_FACTOR


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            Layer(LayerConfig(
                dropout_rate=config.dropout_rate,
                embedding_dim=config.embedding_dim,
                pwff_hidden_dim=config.pwff_hidden_dim,
                num_heads=config.num_heads,
                max_wavelength=config.max_wavelength,
                scale_factor=config.scale_factor,
            ))
            for _ in range(config.num_layers)
        ])

    def forward(self, input: torch.Tensor, positions: torch.Tensor, attn_mask: torch.Tensor):
        x = input
        for layer in self.layers:
            x = layer(x, positions, attn_mask)
        return x
