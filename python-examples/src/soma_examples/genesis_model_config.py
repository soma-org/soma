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

from soma_models.v1.flax import Model, ModelConfig
from flax import nnx

from soma_sdk import Keypair, SomaClient


def generate(seed: int = 42, base_url: str | None = None, out_dir: str = ".") -> str:

    rngs = nnx.Rngs(seed)
    model = Model(ModelConfig(dropout_rate=0.0), rngs=rngs)
    model_bytes = model.save_bytes()

    # 2. Encrypt
    encrypted_bytes, decryption_key = SomaClient.encrypt_weights(model_bytes)

    # 3. Compute fields (all base58)
    checksum = SomaClient.commitment(encrypted_bytes)
    weights_commitment = SomaClient.commitment(model_bytes)
    size = len(encrypted_bytes)

    # 4. Generate owner keypair
    keypair = Keypair.generate()
    owner_address = keypair.address()
    secret_key = keypair.to_secret_key()

    # 5. Create output folder named by weights commitment
    model_dir = os.path.join(out_dir, weights_commitment)
    os.makedirs(model_dir, exist_ok=True)

    # 6. Save encrypted weights file
    enc_filename = f"{checksum}.safetensors.aes256ctr"
    enc_path = os.path.join(model_dir, enc_filename)
    with open(enc_path, "wb") as f:
        f.write(encrypted_bytes)

    if base_url:
        weights_url = f"{base_url}/{enc_filename}"
    else:
        weights_url = f"<UPLOAD {enc_path} AND PASTE URL HERE>"

    # 7. Build YAML and write to file
    yaml = f"""\
- owner: "{owner_address}"
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

    yaml_path = os.path.join(model_dir, "genesis_model_config.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml)

    # 8. Write decryption key to file
    key_path = os.path.join(model_dir, "decryption.key")
    with open(key_path, "w") as f:
        f.write(decryption_key)

    # 9. Write owner secret key to file
    secret_key_path = os.path.join(model_dir, "owner.key")
    with open(secret_key_path, "w") as f:
        f.write(secret_key)

    return model_dir


def main():
    parser = argparse.ArgumentParser(description="Generate GenesisModelConfig YAML")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for weights")
    parser.add_argument("--url", type=str, default=None, help="Weights URL (optional)")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    model_dir = generate(seed=args.seed, base_url=args.url, out_dir=args.out_dir)

    print(f"Model config written to: {model_dir}")


if __name__ == "__main__":
    main()
