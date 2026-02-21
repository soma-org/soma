"""Example: upload data to mock S3, score via the scoring service.

Usage:
    1. Start scoring service:  soma score --small-model
    2. Run:                    uv run soma-example-scoring

The scoring service is a standalone HTTP server that wraps the Soma runtime.
It accepts data/model URLs and a target embedding, runs inference and distance
computation, and returns the results. This script demonstrates the full flow:

  - Spin up a local S3-compatible object store (moto)
  - Generate valid safetensors model weights (small model config)
  - Encrypt and upload model weights + test data
  - Score via the ScoringClient from soma_sdk
  - Print the results that would be passed to ``wallet.submit_data()``

NOTE: This example uses --small-model (embedding_dim=16, num_layers=2) to keep
weights small (~50KB). For production, omit --small-model and use real weights.
"""

import os
import urllib.error

from safetensors.numpy import save as save_safetensors

from soma_sdk import ScoringClient, commitment, encrypt_weights
from soma_sdk.scoring import ModelManifest
from soma_examples.model_utils import EMBEDDING_DIM, build_model_weights
from soma_examples.local_storage import LocalStorage

SCORING_URL = os.environ.get("SCORING_URL", "http://127.0.0.1:9124")


def main():
    client = ScoringClient(SCORING_URL)

    with LocalStorage() as storage:
        # 1. Upload test data
        test_data = os.urandom(1024)
        data_url = storage.upload("data.bin", test_data)
        print(f"Uploaded 1KB test data -> {data_url}")
        print(f"  checksum: {commitment(test_data)}")

        # 2. Generate, encrypt, and upload model weights (small config)
        print("Generating small model weights...")
        model_bytes = save_safetensors(build_model_weights(seed=42))
        encrypted_bytes, decryption_key = encrypt_weights(model_bytes)
        model_url = storage.upload("weights.safetensors.enc", encrypted_bytes)
        print(f"Uploaded model weights ({len(encrypted_bytes)} bytes, encrypted) -> {model_url}")

        # 3. Score via the ScoringClient
        print(f"\nScoring via {SCORING_URL} ...")
        try:
            result = client.score(
                data_url=data_url,
                data=test_data,
                models=[
                    ModelManifest(
                        url=model_url,
                        encrypted_weights=encrypted_bytes,
                        decryption_key=decryption_key,
                    )
                ],
                target_embedding=[0.1] * EMBEDDING_DIM,
            )

            print(f"\nWinner model index: {result.winner}")
            print(f"Distance score: {result.distance}")
            print(f"Embedding dim: {len(result.embedding)}")
            print("\nThese values can be passed to:")
            print(
                "  wallet.submit_data(target_id=..., data=..., data_url=..., "
                "model_id=..., embedding=result.embedding, "
                "distance_score=result.distance[0])"
            )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            print(f"Scoring request failed ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            print(f"Could not connect to scoring service at {SCORING_URL}: {e.reason}")
            print("Make sure 'soma score --small-model' is running first.")


if __name__ == "__main__":
    main()
