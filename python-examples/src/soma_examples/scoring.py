"""Example: upload data to mock S3, score via the scoring service, prepare on-chain submission.

Usage:
    1. Start scoring service:  soma score --small-model
    2. Run:                    uv run soma-example-scoring

The scoring service is a standalone HTTP server that wraps the Soma runtime.
It accepts data/model URLs and a target embedding, runs inference and distance
computation, and returns the results. This script demonstrates the full flow:

  - Spin up a local S3-compatible object store (moto)
  - Generate valid safetensors model weights (small model config)
  - Upload test data and model weights
  - Score via the ScoringClient from soma_sdk
  - Print the results that would be passed to ``build_submit_data()``

NOTE: This example uses --small-model (embedding_dim=16, num_layers=2) to keep
weights small (~50KB). For production, omit --small-model and use real weights.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import boto3
import urllib.request

from arrgen import normal_array
from safetensors.numpy import save_file as save_safetensors
from soma_sdk.scoring import ModelManifest, ScoringClient

SCORING_URL = os.environ.get("SCORING_URL", "http://127.0.0.1:9124")
MOTO_PORT = int(os.environ.get("MOTO_PORT", "5555"))

# Must match the --small-model config in the scoring service.
EMBEDDING_DIM = 16
PWFF_HIDDEN_DIM = 32
NUM_LAYERS = 2
NUM_HEADS = 4
VOCAB_SIZE = 264


def build_model_weights(seed: int) -> dict:
    """Generate valid safetensors weights for the small model config.

    Mirrors the Rust model architecture: embedding + transformer encoder +
    final layer norm + predictor.
    """
    tensors = {}
    for layer in range(NUM_LAYERS):
        lseed = seed + layer
        d, h = EMBEDDING_DIM, PWFF_HIDDEN_DIM
        tensors.update(
            {
                f"encoder.layers.{layer}.norm_1.gamma": normal_array(
                    lseed + 1, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.norm_1.beta": normal_array(
                    lseed + 2, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.query.weight": normal_array(
                    lseed + 3, [d, d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.query.bias": normal_array(
                    lseed + 4, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.key.weight": normal_array(
                    lseed + 5, [d, d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.key.bias": normal_array(
                    lseed + 6, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.value.weight": normal_array(
                    lseed + 7, [d, d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.value.bias": normal_array(
                    lseed + 8, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.output.weight": normal_array(
                    lseed + 9, [d, d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.attention.output.bias": normal_array(
                    lseed + 10, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.norm_2.gamma": normal_array(
                    lseed + 11, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.norm_2.beta": normal_array(
                    lseed + 12, [d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.pwff.linear_inner.weight": normal_array(
                    lseed + 13, [d, h], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.pwff.linear_inner.bias": normal_array(
                    lseed + 14, [h], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.pwff.linear_outer.weight": normal_array(
                    lseed + 15, [h, d], 0.0, 1.0
                ),
                f"encoder.layers.{layer}.pwff.linear_outer.bias": normal_array(
                    lseed + 16, [d], 0.0, 1.0
                ),
            }
        )
    tensors["final_norm.gamma"] = normal_array(seed + 100, [EMBEDDING_DIM], 0.0, 1.0)
    tensors["final_norm.beta"] = normal_array(seed + 200, [EMBEDDING_DIM], 0.0, 1.0)
    tensors["embedding.weight"] = normal_array(
        seed + 250, [VOCAB_SIZE, EMBEDDING_DIM], 0.0, 1.0
    )
    tensors["predictor.weight"] = normal_array(
        seed + 300, [EMBEDDING_DIM, VOCAB_SIZE], 0.0, 1.0
    )
    tensors["predictor.bias"] = normal_array(seed + 400, [VOCAB_SIZE], 0.0, 1.0)
    return tensors


def wait_for_server(url: str, timeout: float = 10.0):
    """Poll a URL until it responds or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url)
            return
        except Exception:
            time.sleep(0.3)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")


def main():
    client = ScoringClient(SCORING_URL)

    # 1. Start moto_server as subprocess (serves S3-compatible API on localhost)
    print(f"Starting moto S3 server on port {MOTO_PORT}...")
    moto_proc = subprocess.Popen(
        [sys.executable, "-m", "moto.server", "-p", str(MOTO_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        endpoint = f"http://127.0.0.1:{MOTO_PORT}"
        wait_for_server(endpoint)
        print(f"Moto S3 server ready at {endpoint}")

        # 2. Create S3 bucket with public-read policy
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
        )
        s3.create_bucket(Bucket="test-bucket")
        s3.put_bucket_policy(
            Bucket="test-bucket",
            Policy=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": "s3:GetObject",
                            "Resource": "arn:aws:s3:::test-bucket/*",
                        }
                    ],
                }
            ),
        )

        # 3. Upload test data
        test_data = os.urandom(1024)
        data_checksum = hashlib.blake2b(test_data, digest_size=32).hexdigest()
        s3.put_object(Bucket="test-bucket", Key="data.bin", Body=test_data)
        data_url = f"{endpoint}/test-bucket/data.bin"
        print(f"Uploaded 1KB test data -> {data_url}")
        print(f"  checksum: {data_checksum}")

        # 4. Generate and upload valid model weights (small config)
        print("Generating small model weights...")
        weights = build_model_weights(seed=42)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_safetensors(weights, f.name)
            model_bytes = open(f.name, "rb").read()
            os.unlink(f.name)

        model_checksum = hashlib.blake2b(model_bytes, digest_size=32).hexdigest()
        s3.put_object(Bucket="test-bucket", Key="weights.safetensors", Body=model_bytes)
        model_url = f"{endpoint}/test-bucket/weights.safetensors"
        print(f"Uploaded model weights ({len(model_bytes)} bytes) -> {model_url}")

        # 5. Score via the ScoringClient
        print(f"\nScoring via {SCORING_URL} ...")
        try:
            result = client.score(
                data_url=data_url,
                data_checksum=data_checksum,
                data_size=len(test_data),
                models=[
                    ModelManifest(
                        url=model_url,
                        checksum=model_checksum,
                        size=len(model_bytes),
                    )
                ],
                target_embedding=[0.1] * EMBEDDING_DIM,
                seed=42,
            )

            print(f"\nWinner model index: {result.winner}")
            print(f"Distance score: {result.distance}")
            print(f"Embedding dim: {len(result.embedding)}")
            print("\nThese values can be passed to:")
            print(
                "  wallet.build_submit_data(sender, target_id, ..., "
                "embedding=result.embedding, "
                "distance_score=result.distance[0])"
            )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            print(f"Scoring request failed ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            print(f"Could not connect to scoring service at {SCORING_URL}: {e.reason}")
            print("Make sure 'soma score --small-model' is running first.")

    finally:
        moto_proc.terminate()
        moto_proc.wait()
        print("\nMoto server stopped.")


if __name__ == "__main__":
    main()
