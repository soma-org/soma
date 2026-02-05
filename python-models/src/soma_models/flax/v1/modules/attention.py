from collections.abc import Callable
import functools
from jax import Array
import jax.numpy as jnp
from flax.nnx.module import Module, first_from
from flax.nnx.nn.attention import dot_product_attention
from flax.typing import (
    Dtype,
    Initializer,
    PrecisionLike,
)
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import (
    LinearGeneral,
    default_kernel_init,
)
from flax.nnx import rnglib


def apply_rope(
    inputs: Array,  # [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM]
    positions: Array,  # [BATCH_SIZE, SEQ_LEN]
    head_dim: int,
    max_wavelength: int = 10_000,
    scale_factor: float = 1.0,
) -> Array:
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = max_wavelength**fraction
    positions = positions[..., jnp.newaxis]
    timescale = timescale[jnp.newaxis, jnp.newaxis, :]  # ty: ignore[non-subscriptable]

    sinusoid_inp = positions / timescale
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
    sinusoid_inp /= scale_factor

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


class MultiHeadAttention(Module):
    def __init__(
        self,
        num_heads: int,
        num_features: int,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        out_kernel_init: Initializer | None = None,
        bias_init: Initializer = initializers.zeros_init(),
        out_bias_init: Initializer | None = None,
        use_bias: bool = True,
        attention_fn: Callable[..., Array] = dot_product_attention,
        max_wavelength: float = 10_000.0,
        scale_factor: float = 1.0,
        rngs: rnglib.Rngs,
        keep_rngs: bool = True,
    ):
        self.num_heads = num_heads
        self.num_features = num_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.broadcast_dropout = broadcast_dropout
        self.dropout_rate = dropout_rate
        self.deterministic = deterministic
        self.precision = precision
        self.kernel_init = kernel_init
        self.out_kernel_init = out_kernel_init
        self.bias_init = bias_init
        self.out_bias_init = out_bias_init
        self.use_bias = use_bias
        self.attention_fn = attention_fn
        self.max_wavelength = max_wavelength
        self.scale_factor = scale_factor

        if self.num_features % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.num_features}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.num_features // self.num_heads

        linear_general = functools.partial(
            LinearGeneral,
            out_features=(self.num_heads, self.head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        self.query = linear_general(self.num_features, rngs=rngs)
        self.key = linear_general(self.num_features, rngs=rngs)
        self.value = linear_general(self.num_features, rngs=rngs)

        self.out = LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.num_features,
            axis=(-2, -1),
            kernel_init=self.out_kernel_init or self.kernel_init,
            bias_init=self.out_bias_init or self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=rngs,
        )
        self.rngs = rngs.dropout.fork() if keep_rngs and dropout_rate > 0 else None

    def __call__(
        self,
        inputs: Array,
        *,
        mask: Array | None = None,
        deterministic: bool | None = None,
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        positions: Array | None = None,
    ):
        if rngs is None:
            rngs = self.rngs
        elif isinstance(rngs, rnglib.Rngs):
            rngs = rngs.dropout

        if inputs.shape[-1] != self.num_features:
            raise ValueError(
                f"Incompatible input dimension, got {inputs.shape[-1]} "
                f"but module expects {self.num_features}."
            )

        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        if positions is not None:
            # Apply RoPE to query and key
            query = apply_rope(
                query,
                positions,
                head_dim=self.head_dim,
                max_wavelength=self.max_wavelength,
                scale_factor=self.scale_factor,
            )
            key = apply_rope(
                key,
                positions,
                head_dim=self.head_dim,
                max_wavelength=self.max_wavelength,
                scale_factor=self.scale_factor,
            )

        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            deterministic = first_from(
                deterministic,
                self.deterministic,
                error_msg="""No `deterministic` argument was provided to MultiHeadAttention
          as either a __call__ argument, class attribute, or nnx.flag.""",
            )
            if not deterministic:
                if rngs is None:
                    raise ValueError(
                        "'rngs' must be provided to __call__ method if "
                        "MultiHeadAttention instance is defined with keep_rngs=False."
                    )
                dropout_rng = rngs()
            else:
                dropout_rng = None
        else:
            deterministic = True
            dropout_rng = None

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
            module=None,
        )
        # back to the original inputs dimensions
        out = self.out(x)
        return out
