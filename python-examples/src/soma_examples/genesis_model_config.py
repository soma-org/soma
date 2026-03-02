"""Generate GenesisModelConfig YAML from model weights.

Builds random model weights (matching the small-model architecture),
encrypts them, computes all commitments/checksums, and prints the YAML
block ready to paste into a genesis config.

Also writes the encrypted weights to a file so you can upload them.

Usage:

    uv run soma-example-genesis-model-config

    # With a custom seed and output directory:
    uv run soma-example-genesis-model-config --seed 99 --out-dir /tmp

    # With a pre-set weights URL:
    uv run soma-example-genesis-model-config --url https://storage.example.com/weights.safetensors.enc
"""

import argparse
import os

from safetensors.numpy import save as save_safetensors

from soma_sdk import SomaClient
from soma_examples.model_utils import build_model_weights


def generate(seed: int = 42, url: str | None = None, out_dir: str = ".") -> str:
    """Return a GenesisModelConfig YAML string and save encrypted weights."""

    # 1. Build model weights
    model_bytes = save_safetensors(build_model_weights(seed=seed))

    # 2. Encrypt
    encrypted_bytes, decryption_key = SomaClient.encrypt_weights(model_bytes)

    # 3. Compute fields (all base58)
    checksum = SomaClient.commitment(encrypted_bytes)
    weights_commitment = SomaClient.commitment(model_bytes)
    size = len(encrypted_bytes)

    # 4. Save encrypted weights file
    enc_path = os.path.join(out_dir, f"weights-seed{seed}.safetensors.enc")
    with open(enc_path, "wb") as f:
        f.write(encrypted_bytes)

    weights_url = url or f"<UPLOAD {enc_path} AND PASTE URL HERE>"

    # 5. Build YAML
    yaml = f"""\
- owner: "<OWNER ADDRESS>"
  manifest:
    V1:
      url: "{weights_url}"
      metadata:
        V1:
          checksum: "{checksum}"
          size: {size}
  decryption_key: "{decryption_key}"
  weights_commitment: "{weights_commitment}"
  architecture_version: 1
  commission_rate: 1000
  initial_stake: 1000000000"""

    return yaml, enc_path


def main():
    parser = argparse.ArgumentParser(description="Generate GenesisModelConfig YAML")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for weights")
    parser.add_argument("--url", type=str, default=None, help="Weights URL (optional)")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    yaml, enc_path = generate(seed=args.seed, url=args.url, out_dir=args.out_dir)

    print(f"Encrypted weights saved to: {enc_path}")
    print()
    print("--- GenesisModelConfig YAML ---")
    print(yaml)


if __name__ == "__main__":
    main()
