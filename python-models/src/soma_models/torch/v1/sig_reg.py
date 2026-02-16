import torch
import torch.nn as nn
from dataclasses import dataclass
from soma_models.config import V1_SIG_REG_T_MAX, V1_SIG_REG_POINTS, V1_SIG_REG_SLICES


@dataclass
class SIGRegConfig:
    t_max: float = V1_SIG_REG_T_MAX
    points: int = V1_SIG_REG_POINTS
    slices: int = V1_SIG_REG_SLICES


class SIGReg(nn.Module):
    def __init__(self, config: SIGRegConfig):
        super().__init__()
        t = torch.linspace(0, config.t_max, config.points)
        dt = config.t_max / (config.points - 1)
        weights = torch.full((config.points,), 2 * dt)
        weights[0] = dt
        weights[-1] = dt
        phi = torch.exp(-0.5 * (t**2))
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        A = noise / torch.norm(noise, dim=0)
        x = x @ A
        N = x.shape[-1]
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = torch.mean(torch.cos(x_t), dim=-2)
        sin_mean = torch.mean(torch.sin(x_t), dim=-2)
        err = ((cos_mean - self.phi) ** 2) + (sin_mean**2)
        integrated = torch.sum(err * self.weights, dim=-1) * N
        return integrated.mean()
