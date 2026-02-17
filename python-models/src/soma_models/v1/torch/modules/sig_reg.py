import torch
import torch.nn as nn

from soma_models.v1.configs import SIGRegConfig


class SIGReg(nn.Module):
    def __init__(self, config: SIGRegConfig):
        super().__init__()
        self.config = config
        t = torch.linspace(0, config.t_max, config.points)
        dt = config.t_max / (config.points - 1)
        weights = torch.full((config.points,), 2 * dt)
        weights[0] = dt
        weights[-1] = dt
        phi = torch.exp(-0.5 * (t**2))
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            x.shape[-1], self.config.slices, dtype=x.dtype, device=x.device
        )
        return self.compute(x, noise)

    def compute(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        A = noise / torch.norm(noise, dim=0)
        x = x @ A
        N = x.shape[-1]
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = torch.mean(torch.cos(x_t), dim=-2)
        sin_mean = torch.mean(torch.sin(x_t), dim=-2)
        err = ((cos_mean - self.phi) ** 2) + (sin_mean**2)
        integrated = torch.sum(err * self.weights, dim=-1) * N
        return integrated.mean() * self.config.coefficient
