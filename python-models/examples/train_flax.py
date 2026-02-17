"""Example: training a small V1 model with Flax (JAX) and optax.

Uses a tiny model config so it runs in seconds on CPU.
The dataset is a short repeated text converted to raw bytes.
"""

import jax.numpy as jnp
import optax
from flax import nnx

from soma_models.v1.configs import ModelConfig, SIGRegConfig
from soma_models.v1.data import prepare_batches
from soma_models.flax.v1.model import Model
from soma_models.flax.v1.sig_reg import SIGReg
from soma_models.flax.v1.loss import compute_loss

# ---------------------------------------------------------------------------
# 1. Dataset — a short repeated sentence converted to raw bytes
# ---------------------------------------------------------------------------
TEXT = "The quick brown fox jumps over the lazy dog. " * 20
DATA = TEXT.encode("utf-8")

SEQ_LEN = 64
BATCH_SIZE = 4

batches = prepare_batches(DATA, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
print(f"Dataset: {len(DATA)} bytes → {sum(b['token_ids'].shape[0] for b in batches)} sequences in {len(batches)} batch(es)")

# ---------------------------------------------------------------------------
# 2. Model — small config for quick iteration
# ---------------------------------------------------------------------------
config = ModelConfig(
    dropout_rate=0.1,
    embedding_dim=32,
    pwff_hidden_dim=64,
    num_layers=2,
    num_heads=4,
    vocab_size=264,
)
rngs = nnx.Rngs(0)
model = Model(config, rngs=rngs)
model.train()
sig_reg_config = SIGRegConfig()
sig_reg = SIGReg(sig_reg_config, rngs=rngs)

tx = optax.adam(1e-3)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------
EPOCHS = 5
SEED = 42


@nnx.jit
def train_step(model, sig_reg, optimizer, token_ids, positions, attn_mask, targets):
    def loss_fn(model):
        return compute_loss(
            model, sig_reg, sig_reg_config,
            token_ids, positions, attn_mask, targets,
            seed=SEED,
        )

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True, argnums=nnx.DiffState(0, nnx.Param))
    (loss, _embedding), grads = grad_fn(model)
    optimizer.update(model, grads)
    return loss


for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n_batches = 0
    for batch in batches:
        token_ids = jnp.array(batch["token_ids"])
        positions = jnp.array(batch["positions"])
        attn_mask = jnp.array(batch["attn_mask"])
        targets = jnp.array(batch["targets"])

        loss = train_step(model, sig_reg, optimizer, token_ids, positions, attn_mask, targets)

        epoch_loss += float(loss)
        n_batches += 1

    print(f"Epoch {epoch + 1}/{EPOCHS}  loss={epoch_loss / n_batches:.4f}")

# ---------------------------------------------------------------------------
# 4. Save weights
# ---------------------------------------------------------------------------
model.save("v1_flax_example.safetensors")
print("Saved weights to v1_flax_example.safetensors")
