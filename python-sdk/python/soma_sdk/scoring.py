"""Scoring service client for the Soma network.

The scoring service (``soma score``) is a standalone HTTP server that wraps the
Soma runtime.  It accepts data/model URLs and a target embedding, runs
inference and distance computation, and returns the results.

Example::

    from soma_sdk.scoring import ScoringClient, ModelManifest

    client = ScoringClient()           # default http://127.0.0.1:9124
    result = client.score(
        data_url="http://...",
        data_checksum="abcd...",
        data_size=1024,
        models=[ModelManifest(url="http://...", checksum="...", size=2048)],
        target_embedding=[0.1] * 2048,
        seed=42,
    )
    print(result.winner, result.distance)
"""

import json
import urllib.request
from dataclasses import dataclass


@dataclass
class ScoreResult:
    """Result returned by the scoring service."""

    winner: int
    loss_score: list[float]
    embedding: list[float]
    distance: list[float]


@dataclass
class ModelManifest:
    """A model manifest to score against."""

    url: str
    checksum: str
    size: int


class ScoringClient:
    """Client for the Soma scoring service (``soma score``).

    Args:
        url: Base URL of the scoring service.
    """

    def __init__(self, url: str = "http://127.0.0.1:9124"):
        self.url = url.rstrip("/")

    def health(self) -> bool:
        """Return ``True`` if the scoring service is reachable."""
        try:
            with urllib.request.urlopen(f"{self.url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    def score(
        self,
        data_url: str,
        data_checksum: str,
        data_size: int,
        models: list[ModelManifest],
        target_embedding: list[float],
        seed: int,
    ) -> ScoreResult:
        """Run a scoring competition and return the result.

        Args:
            data_url: URL where the data blob can be downloaded.
            data_checksum: Hex-encoded blake2b-256 checksum of the data.
            data_size: Size of the data blob in bytes.
            models: List of model manifests to evaluate.
            target_embedding: Target embedding vector.
            seed: Random seed for reproducibility.

        Returns:
            A :class:`ScoreResult` with winner index, loss scores,
            embedding, and distance.

        Raises:
            urllib.error.HTTPError: If the scoring service returns an error.
        """
        body = json.dumps(
            {
                "data_url": data_url,
                "data_checksum": data_checksum,
                "data_size": data_size,
                "model_manifests": [
                    {"url": m.url, "checksum": m.checksum, "size": m.size}
                    for m in models
                ],
                "target_embedding": target_embedding,
                "seed": seed,
            }
        ).encode()

        req = urllib.request.Request(
            f"{self.url}/score",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())

        return ScoreResult(
            winner=result["winner"],
            loss_score=result["loss_score"],
            embedding=result["embedding"],
            distance=result["distance"],
        )
