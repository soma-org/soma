# SOMA Models

<p>
  <a href="https://pypi.org/project/soma-models"><img src="https://img.shields.io/pypi/v/soma-models.svg" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
</p>

Deterministic model architectures for the SOMA network. The Rust runtime in this crate is the **canonical scoring implementation** — Python implementations in [`soma-models`](https://pypi.org/project/soma-models/) (PyTorch and Flax) are numerically identical.

Weights are serialized in [safetensors](https://huggingface.co/docs/safetensors/) format, the canonical weight exchange format between Python and the Rust runtime.

## Python Package

```bash
pip install soma-models[torch]   # PyTorch
pip install soma-models[flax]    # Flax / JAX
pip install soma-models[all]     # Both
```

Requires Python >= 3.11. See [python-models/README.md](../python-models/README.md) for training guides and framework-specific usage.

## Versioning

Model architectures are **versioned**. Each version defines a fixed architecture, hyperparameters, data contract, and scoring function. The on-chain runtime selects the architecture version when evaluating a model, so weights must match the version they were registered with.

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
| `NUM_LAYERS` | 24 | Number of transformer blocks |
| `MAX_SEQ_LEN` | 1024 | Maximum sequence length during on-chain evaluation |
| `PWFF_HIDDEN_DIM` | 8192 | Feed-forward inner dimension (4x embedding_dim) |
| `VOCAB_SIZE` | 264 | 256 byte tokens + 8 special tokens |
| `MAX_WAVELENGTH` | 10,000 | RoPE positional encoding wavelength |
| `SCALE_FACTOR` | 1.0 | RoPE scale factor |
| `BATCH_SIZE` | 16 | Batch size during on-chain evaluation |

### Data Contract

The model operates on **raw bytes**. During on-chain evaluation, data is processed as follows:

- Each byte (0-255) is its own token ID
- Special tokens: **PAD = 256**, **EOS = 257**
- Data is chunked into non-overlapping sequences of `MAX_SEQ_LEN` (1024) bytes
- EOS is only placed on the **final chunk** and only if it is shorter than `MAX_SEQ_LEN` — it occupies the position immediately after the last data byte. If data length is an exact multiple of `MAX_SEQ_LEN`, no EOS is appended
- Any remaining positions after EOS (or after data if no EOS) are filled with PAD
- **Targets** are the input token IDs shifted left by 1 (next-token prediction), with PAD appended as the final target
- **Position IDs** are global byte offsets for data positions. PAD and EOS positions are clamped to the offset of the last data byte + 1 (they do not continue incrementing)
- Sequences are batched in groups of `BATCH_SIZE` (16)

You are free to prepare your training data however you want — different sequence lengths, different batching, different shuffling. But your model will be **scored** using the contract above, so your training should produce weights that perform well under these conditions.

### Scoring (Loss Function)

Models are scored on-chain by the following loss:

```
loss = cross_entropy + sig_reg_loss
```

The model with the **lowest loss** wins. Both components are:

1. **Cross-entropy loss**: Standard next-token prediction loss over the vocabulary. PAD tokens (256) are masked out and do not contribute to the loss.

2. **SIGReg loss**: A Gaussian uniformity regularizer ([LeJEPA](https://arxiv.org/pdf/2511.08544)) that penalizes embedding collapse. It measures how far the embedding distribution deviates from a standard Gaussian by comparing the characteristic function of projected representations against the Gaussian characteristic function.

| SIGReg Parameter | Value |
|------------------|-------|
| `SIG_REG_T_MAX` | 3.0 |
| `SIG_REG_SLICES` | 256 |
| `SIG_REG_POINTS` | 17 |
| `SIG_REG_COEFFICIENT` | 0.02 |

### Weight Serialization

Weights are stored in safetensors format with a canonical key layout. The serde layer handles all framework-specific transformations automatically:

- **LayerNorm**: `weight`/`bias` (torch) ↔ `gamma`/`beta` (safetensors) ↔ `scale`/`bias` (flax)
- **Linear**: Row-major (torch) ↔ column-major (safetensors/flax)
- **Attention**: Split-head (flax) ↔ flat (safetensors/torch)

Weights are cross-compatible — you can save from one framework and load into the other.

## Rust API

```rust
use models::v1::ModelRunner;
use models::v1::modules::model::ModelConfig;

let config = ModelConfig::default();
let runner = ModelRunner::new(config, device, num_workers);
let output = runner.eval(data, weights, seed).await?;
// output.embedding: Tensor<B, 1>
// output.loss: Tensor<B, 1>
```

## License

Apache-2.0
