"""Local S3-compatible storage using moto for testing.

Wraps a moto mock-S3 server so examples and tests can upload/download
objects without touching real cloud storage.

Usage::

    storage = LocalStorage()
    model_url = storage.upload("weights.safetensors", model_bytes)
    data_url  = storage.upload("data.bin", data_bytes)
    # ... use the URLs ...
    storage.close()
"""

import json
import subprocess
import sys
import time

import boto3


class LocalStorage:
    """In-process moto S3 mock with a public-read bucket."""

    def __init__(self, port: int = 5555, bucket: str = "test-bucket"):
        self.port = port
        self.bucket = bucket
        self.endpoint = f"http://127.0.0.1:{port}"

        # Start moto server
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "moto.server", "-p", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._wait_ready()

        # Create bucket with public-read
        self._s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
        )
        self._s3.create_bucket(Bucket=bucket)
        self._s3.put_bucket_policy(
            Bucket=bucket,
            Policy=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{bucket}/*",
                }],
            }),
        )

    def upload(self, key: str, data: bytes) -> str:
        """Upload *data* under *key* and return its public URL."""
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"{self.endpoint}/{self.bucket}/{key}"

    def close(self):
        """Shut down the moto server."""
        self._proc.terminate()
        self._proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------

    def _wait_ready(self, timeout: float = 10.0):
        import urllib.request

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(self.endpoint)
                return
            except Exception:
                time.sleep(0.3)
        raise TimeoutError(
            f"moto server at {self.endpoint} did not start within {timeout}s"
        )
