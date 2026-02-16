import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from soma_models.config import V1_EMBEDDING_DIM, V1_PWFF_HIDDEN_DIM


@dataclass
class PositionWiseFeedForwardConfig:
    dropout_rate: float
    embedding_dim: int = V1_EMBEDDING_DIM
    pwff_hidden_dim: int = V1_PWFF_HIDDEN_DIM


class PositionWiseFeedForward(nn.Module):
    def __init__(self, config: PositionWiseFeedForwardConfig):
        super().__init__()
        self.linear_inner = nn.Linear(config.embedding_dim, config.pwff_hidden_dim)
        self.linear_outer = nn.Linear(config.pwff_hidden_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.linear_inner(x)
        x = F.gelu(x, approximate='none')
        x = self.dropout(x)
        output = self.linear_outer(x)
        return output
