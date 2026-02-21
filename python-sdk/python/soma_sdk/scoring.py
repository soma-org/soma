"""Scoring service client for the Soma network.

The scoring service (``soma score``) is a standalone HTTP server that wraps the
Soma runtime.  It accepts data/model URLs and a target embedding, runs
inference and distance computation, and returns the results.

Example::

    from soma_sdk.scoring import ScoringClient, ModelManifest

    client = ScoringClient()           # default http://127.0.0.1:9124
    result = client.score(
        data_url="http://...",
        data=raw_bytes,
        models=[ModelManifest(url="http://...", encrypted_weights=enc_bytes, decryption_key=key)],
        target_embedding=[0.1] * 2048,
    )
    print(result.winner, result.distance)
"""

import hashlib
import json
import urllib.request


def _blake2b(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=32).hexdigest()


class ScoreResult:
    """Result returned by the scoring service."""

    __slots__ = ("winner", "loss_score", "embedding", "distance")

    def __init__(
        self,
        winner: int,
        loss_score: list[float],
        embedding: list[float],
        distance: list[float],
    ):
        self.winner = winner
        self.loss_score = loss_score
        self.embedding = embedding
        self.distance = distance


class ModelManifest:
    """A model manifest to score against.

    Either pass ``encrypted_weights`` (checksum and size are computed
    automatically) or explicit ``checksum`` + ``size``.
    """

    __slots__ = ("url", "checksum", "size", "decryption_key")

    def __init__(
        self,
        url: str,
        encrypted_weights: bytes | None = None,
        checksum: str | None = None,
        size: int | None = None,
        decryption_key: str | None = None,
    ):
        self.url = url
        if encrypted_weights is not None:
            self.checksum = checksum or _blake2b(encrypted_weights)
            self.size = size if size is not None else len(encrypted_weights)
        else:
            if checksum is None or size is None:
                raise ValueError(
                    "Either encrypted_weights or both checksum and size are required"
                )
            self.checksum = checksum
            self.size = size
        self.decryption_key = decryption_key


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
        models: list[ModelManifest],
        target_embedding: list[float],
        data: bytes | None = None,
        data_checksum: str | None = None,
        data_size: int | None = None,
        seed: int = 0,
    ) -> ScoreResult:
        """Run a scoring competition and return the result.

        Pass ``data`` (raw bytes) to have checksum and size computed
        automatically, or provide explicit ``data_checksum`` + ``data_size``.
        """
        if data is not None:
            data_checksum = data_checksum or _blake2b(data)
            data_size = data_size if data_size is not None else len(data)
        elif data_checksum is None or data_size is None:
            raise ValueError(
                "Either data or both data_checksum and data_size are required"
            )

        body = json.dumps(
            {
                "data_url": data_url,
                "data_checksum": data_checksum,
                "data_size": data_size,
                "model_manifests": [
                    {
                        "url": m.url,
                        "checksum": m.checksum,
                        "size": m.size,
                        **({"decryption_key": m.decryption_key} if m.decryption_key else {}),
                    }
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
