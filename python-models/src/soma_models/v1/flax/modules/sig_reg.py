import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from soma_models.v1.configs import SIGRegConfig


class SIGReg(nnx.Module):
    def __init__(self, config: SIGRegConfig, rngs: nnx.Rngs):
        self.config = config
        self.rngs = rngs
        t = jnp.linspace(0, config.t_max, config.points)
        dt = config.t_max / (config.points - 1)
        weights = jnp.full((config.points,), 2 * dt)
        weights = weights.at[0].set(dt)
        weights = weights.at[-1].set(dt)
        self.t = t
        self.phi = jnp.exp(-0.5 * (t**2))
        self.weights = weights * self.phi

    def __call__(self, x: Array) -> Array:
        noise = jax.random.normal(
            self.rngs.noise(), shape=(x.shape[-1], self.config.slices), dtype=x.dtype
        )
        return self.compute(x, noise)

    def compute(self, x: Array, noise: Array) -> Array:
        A = noise / jnp.linalg.norm(noise, axis=0)
        x = x @ A
        N = x.shape[-1]
        x_t = x[..., None] * self.t
        cos_mean = jnp.mean(jnp.cos(x_t), axis=-2)
        sin_mean = jnp.mean(jnp.sin(x_t), axis=-2)
        err = ((cos_mean - self.phi) ** 2) + (sin_mean**2)
        integrated = jnp.sum(err * self.weights, axis=-1) * N
        return integrated.mean() * self.config.coefficient
