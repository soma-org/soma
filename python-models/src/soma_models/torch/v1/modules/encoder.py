import torch
import torch.nn as nn

from soma_models.v1.configs import EncoderConfig, LayerConfig
from soma_models.torch.v1.modules.layer import Layer


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
