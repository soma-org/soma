import torch.nn as nn
import torch.nn.functional as F

from soma_models.v1.configs import PositionWiseFeedForwardConfig


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
