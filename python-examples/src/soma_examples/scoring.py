"""Example: upload data to mock S3, score via the scoring service.

Usage:
    1. Start scoring service:  soma score --small-model
    2. Run:                    uv run soma-example-scoring

The scoring service is a standalone gRPC server that wraps the Soma runtime.
It accepts data/model URLs and a target embedding, runs inference and distance
computation, and returns the results. This script demonstrates the full flow:

  - Spin up a local S3-compatible object store (moto)
  - Generate valid safetensors model weights (small model config)
  - Encrypt and upload model weights + test data
  - Score via SomaClient's built-in gRPC scoring client
  - Print the results that would be passed to ``client.submit_data()``

NOTE: This example uses --small-model (embedding_dim=16, num_layers=2) to keep
weights small (~50KB). For production, omit --small-model and use real weights.
"""

import asyncio
import os

from safetensors.numpy import save as save_safetensors

from soma_sdk import SomaClient
from soma_examples.model_utils import EMBEDDING_DIM, build_model_weights
from soma_examples.local_storage import LocalStorage

SCORING_URL = os.environ.get("SCORING_URL", "http://127.0.0.1:9124")


async def run():
    client = await SomaClient("http://127.0.0.1:9000", scoring_url=SCORING_URL)

    with LocalStorage() as storage:
        # 1. Upload test data
        test_data = os.urandom(1024)
        data_url = storage.upload("data.bin", test_data)
        print(f"Uploaded 1KB test data -> {data_url}")
        print(f"  checksum: {SomaClient.commitment(test_data)}")

        # 2. Generate, encrypt, and upload model weights (small config)
        print("Generating small model weights...")
        model_bytes = save_safetensors(build_model_weights(seed=42))
        encrypted_bytes, decryption_key = SomaClient.encrypt_weights(model_bytes)
        model_url = storage.upload("weights.safetensors.enc", encrypted_bytes)
        print(f"Uploaded model weights ({len(encrypted_bytes)} bytes, encrypted) -> {model_url}")

        # 3. Score via SomaClient's gRPC scoring
        print(f"\nScoring via {SCORING_URL} ...")
        try:
            # Build a model manifest as a SimpleNamespace-compatible dict
            from types import SimpleNamespace
            manifest = SimpleNamespace(
                url=model_url,
                encrypted_weights=encrypted_bytes,
                decryption_key=decryption_key,
            )

            result = await client.score(
                data_url=data_url,
                models=[manifest],
                target_embedding=[0.1] * EMBEDDING_DIM,
                data=test_data,
                seed=0,
            )

            print(f"\nWinner model index: {result.winner}")
            print(f"Distance score: {result.distance}")
            print(f"Embedding dim: {len(result.embedding)}")
            print("\nThese values can be passed to:")
            print(
                "  await client.submit_data(signer=kp, target_id=..., data=..., "
                "data_url=..., model_id=..., embedding=result.embedding, "
                "distance_score=result.distance[0])"
            )
        except Exception as e:
            print(f"Scoring request failed: {e}")
            print("Make sure 'soma score --small-model' is running first.")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
