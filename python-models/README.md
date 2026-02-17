# Soma Models

Python implementations of Soma network models. These implementations are **numerically identical** to the Rust runtime — weights trained in Python produce the same outputs when evaluated on-chain.

Both [PyTorch](https://pytorch.org/) and [Flax](https://flax.readthedocs.io/) (JAX) are supported as first-class frameworks. Models are serialized to [safetensors](https://huggingface.co/docs/safetensors/) format, which is the canonical weight exchange format between Python and the Rust runtime.

## Install

```bash
# PyTorch
uv add soma-models[torch]

# Flax / JAX
uv add soma-models[flax]

# Both
uv add soma-models[all]
```

Or with pip:

```bash
pip install soma-models[torch]   # PyTorch
pip install soma-models[flax]    # Flax / JAX
pip install soma-models[all]     # Both
```

## Versioning

Model architectures are **versioned**. Each version defines a fixed architecture, hyperparameters, data contract, and scoring function. The on-chain runtime selects the architecture version when evaluating a model, so your weights must match the version you registered with.

New versions may be introduced via protocol upgrades. Previous versions continue to work for models registered under them.

---

## V1

V1 is a **pre-norm byte-level transformer**. It operates directly on raw bytes — no external tokenizer is needed. The model uses rotary positional embeddings (RoPE), GELU activations, and a next-token prediction objective with a Gaussian uniformity regularizer (SIGReg) to prevent embedding collapse.

### Architecture

```
Input bytes (0–255)
    │
    ▼
Embedding (vocab_size → embedding_dim)
    │
    ▼
Encoder (num_layers × TransformerBlock)
    │   ┌─────────────────────────────────┐
    │   │  Pre-Norm (LayerNorm)           │
    │   │  Multi-Head Attention (RoPE)    │
    │   │  Dropout + Residual             │
    │   │  Pre-Norm (LayerNorm)           │
    │   │  Feed-Forward (GELU)            │
    │   │  Dropout + Residual             │
    │   └─────────────────────────────────┘
    │
    ▼
Final LayerNorm → representations (used for embedding + loss)
    │
    ▼
Linear predictor → logits (used for cross-entropy loss)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EMBEDDING_DIM` | 2048 | Dimension of token embeddings and hidden states |
| `NUM_HEADS` | 8 | Number of attention heads (head_dim = 256) |
| `NUM_LAYERS` | 32 | Number of transformer blocks |
| `MAX_SEQ_LEN` | 8192 | Maximum sequence length during on-chain evaluation |
| `PWFF_HIDDEN_DIM` | 8192 | Feed-forward inner dimension (4 × embedding_dim) |
| `VOCAB_SIZE` | 264 | 256 byte tokens + 8 special tokens |
| `MAX_WAVELENGTH` | 10,000 | RoPE positional encoding wavelength |
| `SCALE_FACTOR` | 1.0 | RoPE scale factor |
| `BATCH_SIZE` | 32 | Batch size during on-chain evaluation |

### Data Contract

The model operates on **raw bytes**. During on-chain evaluation, data is processed as follows:

- Each byte (0–255) is its own token ID
- Special tokens: **PAD = 256**, **EOS = 257**
- Data is chunked into non-overlapping sequences of `MAX_SEQ_LEN` (8192) bytes
- EOS is only placed on the **final chunk** and only if it is shorter than `MAX_SEQ_LEN` — it occupies the position immediately after the last data byte. If data length is an exact multiple of `MAX_SEQ_LEN`, no EOS is appended
- Any remaining positions after EOS (or after data if no EOS) are filled with PAD
- **Targets** are the input token IDs shifted left by 1 (next-token prediction), with PAD appended as the final target
- **Position IDs** are global byte offsets for data positions. PAD and EOS positions are clamped to the offset of the last data byte + 1 (they do not continue incrementing)
- Sequences are batched in groups of `BATCH_SIZE` (32)

You are free to prepare your training data however you want — different sequence lengths, different batching, different shuffling. But your model will be **scored** using the contract above, so your training should produce weights that perform well under these conditions.

### Scoring (Loss Function)

Models are scored on-chain by the following loss:

```
loss = cross_entropy + SIG_REG_COEFFICIENT * sig_reg_loss
```

The model with the **lowest loss** wins. Both components are:

1. **Cross-entropy loss**: Standard next-token prediction loss over the vocabulary. PAD tokens (256) are masked out and do not contribute to the loss.

2. **SIGReg loss**: A Gaussian uniformity regularizer ([LeJEPA](https://arxiv.org/pdf/2511.08544)) that penalizes embedding collapse. It measures how far the embedding distribution deviates from a standard Gaussian by comparing the characteristic function of projected representations against the Gaussian characteristic function.

| SIGReg Parameter | Value |
|------------------|-------|
| `SIG_REG_T_MAX` | 3.0 |
| `SIG_REG_SLICES` | 1024 |
| `SIG_REG_POINTS` | 17 |
| `SIG_REG_COEFFICIENT` | 1.0 |

SIGReg noise is generated deterministically using [arrgen](https://github.com/soma-org/soma/tree/main/arrgen) with the evaluation seed. This means the same weights + data + seed always produce the same score.

### Usage

Both frameworks expose the same API: a `Model`, a `SIGReg` regularizer, and a `compute_loss` function.

#### PyTorch

```python
import torch
from soma_models.torch.v1.model import Model, ModelConfig
from soma_models.torch.v1.sig_reg import SIGReg, SIGRegConfig
from soma_models.torch.v1.loss import compute_loss

# Initialize
model = Model(ModelConfig(dropout_rate=0.1))
sig_reg_config = SIGRegConfig()
sig_reg = SIGReg(sig_reg_config)

# Forward + loss (differentiable)
loss, embedding = compute_loss(
    model, sig_reg, sig_reg_config,
    token_ids=token_ids,         # [batch, seq] int tensor
    positions=positions,         # [batch, seq] int tensor
    attn_mask=attn_mask,         # [batch, 1, seq, seq] bool tensor (causal)
    targets=targets,             # [batch, seq] int tensor (input shifted left by 1)
    seed=42,
)
loss.backward()

# Save / load weights
model.save("weights.safetensors")
model = Model.load("weights.safetensors", ModelConfig(dropout_rate=0.0))
```

#### Flax

```python
import jax.numpy as jnp
from flax import nnx
from soma_models.flax.v1.model import Model, ModelConfig
from soma_models.flax.v1.sig_reg import SIGReg, SIGRegConfig
from soma_models.flax.v1.loss import compute_loss

# Initialize
rngs = nnx.Rngs(0)
model = Model(ModelConfig(dropout_rate=0.1), rngs=rngs)
sig_reg_config = SIGRegConfig()
sig_reg = SIGReg(sig_reg_config, rngs=rngs)

# Forward + loss (differentiable via jax.grad)
loss, embedding = compute_loss(
    model, sig_reg, sig_reg_config,
    token_ids=token_ids,         # [batch, seq] jnp array
    positions=positions,         # [batch, seq] jnp array
    attn_mask=attn_mask,         # [batch, 1, seq, seq] bool array (causal)
    targets=targets,             # [batch, seq] jnp array (input shifted left by 1)
    seed=42,
)

# Save / load weights
model.save("weights.safetensors")
model = Model.load("weights.safetensors", ModelConfig(dropout_rate=0.0), rngs=rngs)
```

### Weight Serialization

Weights are stored in safetensors format with a canonical key layout. The serde layer handles all framework-specific transformations automatically:

- **LayerNorm**: `weight`/`bias` (torch) ↔ `gamma`/`beta` (safetensors) ↔ `scale`/`bias` (flax)
- **Linear**: Row-major (torch) ↔ column-major (safetensors/flax)
- **Attention**: Split-head (flax) ↔ flat (safetensors/torch)

#### PyTorch

```python
from soma_models.torch.v1.model import Model, ModelConfig

# Save
model.save("weights.safetensors")

# Load
model = Model.load("weights.safetensors", ModelConfig(dropout_rate=0.0))
```

#### Flax

```python
from soma_models.flax.v1.model import Model, ModelConfig
from flax import nnx

# Save
model.save("weights.safetensors")

# Load
model = Model.load("weights.safetensors", ModelConfig(dropout_rate=0.0), rngs=nnx.Rngs(0))
```

Weights are cross-compatible — you can save from one framework and load into the other.
