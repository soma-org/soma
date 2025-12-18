import jax.numpy as jnp
from flax import nnx
from jax import random
from jax import Array
from dataclasses import dataclass

from soma_probes.config import (
    V1_SIG_REG_T_MAX,
    V1_SIG_REG_POINTS,
    V1_SIG_REG_SLICES,
)


@dataclass
class SIGRegConfig:
    t_max: float = V1_SIG_REG_T_MAX
    points: int = V1_SIG_REG_POINTS
    slices: int = V1_SIG_REG_SLICES


class SIGReg(nnx.Module):
    def __init__(self, config: SIGRegConfig, rngs: nnx.Rngs):
        t = jnp.linspace(0, config.t_max, config.points)
        dt = config.t_max / (config.points - 1)
        weights = jnp.full((config.points,), 2 * dt)
        weights = weights.at[0].set(dt)
        weights = weights.at[-1].set(dt)
        self.t = t
        self.phi = jnp.exp(-0.5 * (t**2))
        self.weights = weights * self.phi

    def __call__(self, x: Array, noise: Array) -> Array:
        A = noise / jnp.linalg.norm(noise, axis=0)
        x = x @ A
        N = x.shape[-1]
        x_t = x[..., None] * self.t
        cos_mean = jnp.mean(jnp.cos(x_t), axis=-2)
        sin_mean = jnp.mean(jnp.sin(x_t), axis=-2)
        err = ((cos_mean - self.phi) ** 2) + (sin_mean**2)
        integrated = jnp.sum(err * self.weights, axis=-1) * N
        return integrated.mean()
