"""Example: training a small V1 model with PyTorch.

Uses a tiny model config so it runs in seconds on CPU.
The dataset is a short repeated text converted to raw bytes.
"""

import torch
from soma_models.v1.configs import ModelConfig, SIGRegConfig
from soma_models.v1.data import prepare_batches
from soma_models.torch.v1.model import Model
from soma_models.torch.v1.sig_reg import SIGReg
from soma_models.torch.v1.loss import compute_loss

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
model = Model(config)
sig_reg_config = SIGRegConfig()
sig_reg = SIGReg(sig_reg_config)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------
EPOCHS = 5
SEED = 42

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n_batches = 0
    for batch in batches:
        token_ids = torch.tensor(batch["token_ids"])
        positions = torch.tensor(batch["positions"])
        attn_mask = torch.tensor(batch["attn_mask"])
        targets = torch.tensor(batch["targets"])

        loss, _embedding = compute_loss(
            model, sig_reg, sig_reg_config,
            token_ids, positions, attn_mask, targets,
            seed=SEED,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    print(f"Epoch {epoch + 1}/{EPOCHS}  loss={epoch_loss / n_batches:.4f}")

# ---------------------------------------------------------------------------
# 4. Save weights
# ---------------------------------------------------------------------------
model.save("v1_torch_example.safetensors")
print("Saved weights to v1_torch_example.safetensors")
