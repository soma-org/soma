from collections.abc import Callable
from typing import Any
import functools
from jax import Array
from jax import lax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.module import Module, first_from
from flax.nnx.nn.normalization import LayerNorm
from flax.nnx.nn.attention import dot_product_attention, combine_masks
from flax.typing import (
    Dtype,
    Shape,
    Initializer,
    PrecisionLike,
    DotGeneralT,
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
    """Applies RoPE."""
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
    """Multi-head attention.

    Example usage::

      >>> from flax import nnx
      >>> import jax

      >>> layer = nnx.MultiHeadAttention(num_heads=8, in_features=5, qkv_features=16,
      ...                                decode=False, rngs=nnx.Rngs(0))
      >>> key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
      >>> shape = (4, 3, 2, 5)
      >>> q, k, v = (
      ...   jax.random.uniform(key1, shape),
      ...   jax.random.uniform(key2, shape),
      ...   jax.random.uniform(key3, shape),
      ... )

      >>> # different inputs for inputs_q, inputs_k and inputs_v
      >>> out = layer(q, k, v)
      >>> # equivalent output when inferring v
      >>> assert (layer(q, k) == layer(q, k, k)).all()
      >>> # equivalent output when inferring k and v
      >>> assert (layer(q) == layer(q, q)).all()
      >>> assert (layer(q) == layer(q, q, q)).all()

    Args:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      in_features: int or tuple with number of input features.
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection.
      in_kv_features: number of input features for computing key and value.
      dtype: the dtype of the computation (default: infer from inputs and params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      out_kernel_init: optional initializer for the kernel of the output Dense layer,
        if None, the kernel_init is used.
      bias_init: initializer for the bias of the Dense layers.
      out_bias_init: optional initializer for the bias of the output Dense layer,
        if None, the bias_init is used.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts query,
        key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
        num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
      normalize_qk: should QK normalization be applied (arxiv.org/abs/2302.05442).
      rngs: rng key.
      keep_rngs: whether to store the input rngs as attribute (i.e. `self.rngs = rngs`)
        (default: True). If rngs is stored, we should split the module as
        `graphdef, params, nondiff = nnx.split(module, nnx.Param, ...)` where `nondiff`
        contains RNG object associated with stored `self.rngs`.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: int | None = None,
        out_features: int | None = None,
        in_kv_features: int | None = None,
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
        decode: bool | None = None,
        normalize_qk: bool = False,
        max_wavelength: int = 10_000,
        scale_factor: float = 1.0,
        # Deprecated, will be removed.
        qkv_dot_general: DotGeneralT | None = None,
        out_dot_general: DotGeneralT | None = None,
        qkv_dot_general_cls: Any = None,
        out_dot_general_cls: Any = None,
        rngs: rnglib.Rngs,
        keep_rngs: bool = True,
    ):
        self.num_heads = num_heads
        self.in_features = in_features
        self.qkv_features = qkv_features if qkv_features is not None else in_features
        self.out_features = out_features if out_features is not None else in_features
        self.in_kv_features = (
            in_kv_features if in_kv_features is not None else in_features
        )
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
        self.decode = decode
        self.normalize_qk = normalize_qk
        self.max_wavelength = max_wavelength
        self.scale_factor = scale_factor
        self.qkv_dot_general = qkv_dot_general
        self.out_dot_general = out_dot_general
        self.qkv_dot_general_cls = qkv_dot_general_cls
        self.out_dot_general_cls = out_dot_general_cls

        if self.qkv_features % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.qkv_features}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.qkv_features // self.num_heads

        linear_general = functools.partial(
            LinearGeneral,
            out_features=(self.num_heads, self.head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
            dot_general_cls=self.qkv_dot_general_cls,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        self.query = linear_general(self.in_features, rngs=rngs)
        self.key = linear_general(self.in_kv_features, rngs=rngs)
        self.value = linear_general(self.in_kv_features, rngs=rngs)

        self.query_ln: LayerNorm | None
        self.key_ln: LayerNorm | None
        if self.normalize_qk:
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            self.query_ln = LayerNorm(
                self.head_dim,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
            self.key_ln = LayerNorm(
                self.head_dim,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self.query_ln = None
            self.key_ln = None

        self.out = LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.out_features,
            axis=(-2, -1),
            kernel_init=self.out_kernel_init or self.kernel_init,
            bias_init=self.out_bias_init or self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=self.out_dot_general,
            dot_general_cls=self.out_dot_general_cls,
            rngs=rngs,
        )
        self.rngs = rngs.dropout.fork() if keep_rngs and dropout_rate > 0 else None

        self.cached_key: nnx.Cache[Array] | None = None
        self.cached_value: nnx.Cache[Array] | None = None
        self.cache_index: nnx.Cache[Array] | None = None

    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Array | None = None,
        inputs_v: Array | None = None,
        *,
        mask: Array | None = None,
        deterministic: bool | None = None,
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        sow_weights: bool = False,
        decode: bool | None = None,
        positions: Array | None = None,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        If both inputs_k and inputs_v are None, they will both copy the value of
        inputs_q (self attention).
        If only inputs_v is None, it will copy the value of inputs_k.

        Args:
          inputs_q: input queries of shape `[batch_sizes..., length, features]`.
          inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
            inputs_k will copy the value of inputs_q.
          inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
            inputs_v will copy the value of inputs_k.
          mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
            key/value_length]`. Attention weights are masked out if their
            corresponding mask value is `False`.
          deterministic: if false, the attention weight is masked randomly using
            dropout, whereas if true, the attention weights are deterministic. The
            ``deterministic`` flag passed into the call method will take precedence
            over the ``deterministic`` flag passed into the constructor.
          rngs: rng key. The rng key passed into the call method will take
            precedence over the rng key passed into the constructor.
          sow_weights: if ``True``, the attention weights are sowed into the
            'intermediates' collection.
          decode: whether to prepare and use an autoregressive cache. The ``decode``
            flag passed into the call method will take precedence over the ``decode``
            flag passed into the constructor.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        if rngs is None:
            rngs = self.rngs
        elif isinstance(rngs, rnglib.Rngs):
            rngs = rngs.dropout

        if inputs_k is None:
            if inputs_v is not None:
                raise ValueError(
                    "`inputs_k` cannot be None if `inputs_v` is not None. "
                    "To have both `inputs_k` and `inputs_v` be the same value, pass in the "
                    "value to `inputs_k` and leave `inputs_v` as None."
                )
            inputs_k = inputs_q
        if inputs_v is None:
            inputs_v = inputs_k

        if inputs_q.shape[-1] != self.in_features:
            raise ValueError(
                f"Incompatible input dimension, got {inputs_q.shape[-1]} "
                f"but module expects {self.in_features}."
            )

        query = self.query(inputs_q)
        key = self.key(inputs_k)
        value = self.value(inputs_v)

        if self.normalize_qk:
            assert self.query_ln is not None and self.key_ln is not None
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = self.query_ln(query)
            key = self.key_ln(key)

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

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        decode = first_from(
            decode,
            self.decode,
            error_msg="""No `decode` argument was provided to MultiHeadAttention
        as either a __call__ argument, class attribute, or nnx.flag.""",
        )

        if decode:
            if (
                self.cached_key is None
                or self.cached_value is None
                or self.cache_index is None
            ):
                raise ValueError(
                    "Autoregressive cache not initialized, call ``init_cache`` first."
                )
            (
                *batch_dims,
                max_length,
                num_heads,
                depth_per_head,
            ) = self.cached_key.value.shape
            # shape check of cached keys against query input
            expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
            if expected_shape != query.shape:
                raise ValueError(
                    "Autoregressive cache shape error, "
                    "expected query shape %s instead got %s."
                    % (expected_shape, query.shape)
                )
            # update key, value caches with our new 1d spatial slices
            cur_index = self.cache_index[...]
            zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
            indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
            key = lax.dynamic_update_slice(self.cached_key[...], key, indices)
            value = lax.dynamic_update_slice(self.cached_value[...], value, indices)
            self.cached_key[...] = key
            self.cached_value[...] = value
            self.cache_index[...] += 1
            # causal mask for cached decoder self-attention:
            # our single query position should only attend to those key
            # positions that have already been generated and cached,
            # not the remaining zero elements.
            mask = combine_masks(
                mask,
                jnp.broadcast_to(
                    jnp.arange(max_length) <= cur_index,
                    tuple(batch_dims) + (1, 1, max_length),
                ),
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
            module=self if sow_weights else None,
        )
        # back to the original inputs dimensions
        out = self.out(x)
        return out

    def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32):
        """Initializes cache for fast autoregressive decoding. When
        ``decode=True``, this method must be called first before performing
        forward inference. When in decode mode, only one token must be passed
        at a time.

        Example usage::

          >>> from flax import nnx
          >>> import jax.numpy as jnp
          ...
          >>> batch_size = 5
          >>> embed_dim = 3
          >>> x = jnp.ones((batch_size, 1, embed_dim)) # single token
          ...
          >>> model_nnx = nnx.MultiHeadAttention(
          ...   num_heads=2,
          ...   in_features=3,
          ...   qkv_features=6,
          ...   out_features=6,
          ...   decode=True,
          ...   rngs=nnx.Rngs(42),
          ... )
          ...
          >>> # out_nnx = model_nnx(x)  <-- throws an error because cache isn't initialized
          ...
          >>> model_nnx.init_cache(x.shape)
          >>> out_nnx = model_nnx(x)
        """
        cache_shape = (*input_shape[:-1], self.num_heads, self.head_dim)
        self.cached_key = nnx.Cache(jnp.zeros(cache_shape, dtype))
        self.cached_value = nnx.Cache(jnp.zeros(cache_shape, dtype))
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))
