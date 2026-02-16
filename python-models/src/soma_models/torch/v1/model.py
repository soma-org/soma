import torch
import torch.nn as nn

from soma_models.v1.configs import ModelConfig, EncoderConfig
from soma_models.torch.v1.modules.encoder import Encoder
from soma_models.torch.serde import Serializable


class Model(Serializable, nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.encoder = Encoder(
            EncoderConfig(
                dropout_rate=config.dropout_rate,
                embedding_dim=config.embedding_dim,
                pwff_hidden_dim=config.pwff_hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                max_wavelength=config.max_wavelength,
                scale_factor=config.scale_factor,
            )
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim, eps=1e-5)
        self.predictor = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(
        self,
        input: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(input)
        x = self.encoder(x, positions, attn_mask)
        x = self.final_norm(x)
        return x

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.predictor(embeddings)
