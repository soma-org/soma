"""Soma Python SDK — interact with the Soma network."""

import hashlib
import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen

from soma_sdk.soma_sdk import SomaClient, WalletContext
from soma_sdk.soma_sdk import encrypt_weights as _encrypt_weights_rs
from soma_sdk.scoring import ModelManifest, ScoreResult, ScoringClient

# ---------------------------------------------------------------------------
# Constants & unit helpers
# ---------------------------------------------------------------------------

SHANNONS_PER_SOMA = 1_000_000_000


def to_shannons(soma: float) -> int:
    """Convert SOMA to shannons (the smallest on-chain unit)."""
    return int(soma * SHANNONS_PER_SOMA)


def to_soma(shannons: int) -> float:
    """Convert shannons to SOMA."""
    return shannons / SHANNONS_PER_SOMA


# ---------------------------------------------------------------------------
# Cryptographic helpers
# ---------------------------------------------------------------------------

def blake2b_commitment(data: bytes) -> str:
    """Compute blake2b-256 hash, return 64-char hex string."""
    return hashlib.blake2b(data, digest_size=32).hexdigest()


# Alias: callers shouldn't need to know the hash algorithm.
commitment = blake2b_commitment


def random_object_id() -> str:
    """Generate random 32-byte ObjectID as 0x-prefixed hex."""
    return "0x" + os.urandom(32).hex()


def encrypt_weights(data: bytes, key: bytes | None = None) -> tuple[bytes, str]:
    """Encrypt model weights with AES-256-CTR (zero IV).

    Returns ``(encrypted_bytes, key_hex)`` where *key_hex* is the 64-char hex
    string to pass as ``decryption_key`` in ``wallet.reveal_model()``.

    If *key* is ``None`` a random 32-byte key is generated.
    """
    return _encrypt_weights_rs(data, key)


# ---------------------------------------------------------------------------
# Target dataclass — typed attribute access for RPC target responses
# ---------------------------------------------------------------------------

@dataclass
class Target:
    """A mining target returned by ``wallet.get_targets()``."""

    id: str
    status: str
    embedding: list[float]
    model_ids: list[str]
    distance_threshold: float
    reward_pool: int
    generation_epoch: int
    bond_amount: int = 0
    miner: str | None = None
    winning_model_id: str | None = None

    @classmethod
    def _from_json(cls, d: dict) -> "Target":
        def _hex(v: str | None) -> str | None:
            return None if v is None else (v if v.startswith("0x") else f"0x{v}")

        return cls(
            id=_hex(d["id"]),
            status=d.get("status", ""),
            embedding=d.get("embedding", []),
            model_ids=[_hex(m) for m in d.get("modelIds", [])],
            distance_threshold=float(d.get("distanceThreshold", 0)),
            reward_pool=int(d.get("rewardPool", 0)),
            generation_epoch=int(d.get("generationEpoch", 0)),
            bond_amount=int(d.get("bondAmount", 0)),
            miner=_hex(d.get("miner")),
            winning_model_id=_hex(d.get("winningModelId")),
        )


# ---------------------------------------------------------------------------
# Wallet — high-level wrapper with auto-sender/gas and build+execute
# ---------------------------------------------------------------------------

class Wallet:
    """High-level wallet that remembers the active sender and auto-selects gas.

    Wraps ``WalletContext`` and ``SomaClient`` with convenience methods that
    build, sign, and execute transactions in a single call. All amounts are
    denominated in **SOMA** (not shannons).
    """

    def __init__(self, config_path: str, client: "SomaClient"):
        self._ctx = WalletContext(os.path.expanduser(config_path))
        self._client = client
        self._sender: str | None = None

    @property
    def ctx(self) -> WalletContext:
        """Access the underlying ``WalletContext`` for low-level operations."""
        return self._ctx

    @property
    def client(self) -> "SomaClient":
        """Access the underlying ``SomaClient``."""
        return self._client

    @property
    def sender(self) -> str:
        if self._sender is None:
            raise RuntimeError("Call active_address() first")
        return self._sender

    async def active_address(self) -> str:
        """Get the active address and cache it for subsequent calls."""
        self._sender = await self._ctx.active_address()
        return self._sender

    async def get_balance(self) -> float:
        """Get balance for the active sender, in SOMA."""
        shannons = await self._client.get_balance(self.sender)
        return shannons / SHANNONS_PER_SOMA

    async def embedding_dim(self) -> int:
        """Get the target embedding dimension from the network."""
        return await self._client.get_embedding_dim()

    async def get_targets(
        self,
        status: str | None = None,
        epoch: int | None = None,
        limit: int | None = None,
    ) -> list[Target]:
        """List targets as typed ``Target`` objects."""
        raw = json.loads(
            await self._client.list_targets(status=status, epoch=epoch, limit=limit)
        )
        return [Target._from_json(t) for t in raw.get("targets", [])]

    async def get_model_manifests(self, target: Target) -> list[ModelManifest]:
        """Fetch revealed model manifests for a target's models from on-chain state.

        Returns a :class:`ModelManifest` for each model that has been revealed,
        ready to pass directly to ``ScoringClient.score()``.
        """
        raw = json.loads(
            await self._client.get_model_manifests(target.model_ids)
        )
        return [
            ModelManifest(
                url=m["url"],
                checksum=m["checksum"],
                size=m["size"],
                decryption_key=m["decryption_key"],
            )
            for m in raw
        ]

    # -- Model lifecycle ---------------------------------------------------

    async def commit_model(
        self,
        *,
        weights_url: str,
        encrypted_weights: bytes,
        commission_rate: int,
        stake_amount: float | None = None,
        model_id: str | None = None,
        staking_pool_id: str | None = None,
    ) -> str:
        """Build + execute CommitModel. Returns the ``model_id``.

        Args:
            weights_url: URL where the encrypted weights will be hosted.
            encrypted_weights: The encrypted weight bytes (commitments computed
                automatically).
            commission_rate: Commission rate in basis points (e.g. 1000 = 10%).
            stake_amount: Stake in SOMA. Defaults to the on-chain minimum.
            model_id: Explicit model ID, or auto-generated.
            staking_pool_id: Explicit staking pool ID, or auto-generated.
        """
        if stake_amount is None:
            stake_shannons = await self._client.get_model_min_stake()
        else:
            stake_shannons = to_shannons(stake_amount)

        model_id = model_id or random_object_id()
        staking_pool_id = staking_pool_id or random_object_id()
        tx_data = await self._ctx.build_commit_model(
            sender=self.sender,
            model_id=model_id,
            weights_url_commitment=commitment(weights_url.encode()),
            weights_commitment=commitment(encrypted_weights),
            stake_amount=stake_shannons,
            commission_rate=commission_rate,
            staking_pool_id=staking_pool_id,
        )
        await self._ctx.execute(tx_data, "CommitModel")
        return model_id

    async def reveal_model(
        self,
        *,
        model_id: str,
        weights_url: str,
        encrypted_weights: bytes,
        decryption_key: str,
        embedding: list[float],
    ) -> None:
        """Build + execute RevealModel.

        Checksum and size are computed from *encrypted_weights* automatically.
        """
        tx_data = await self._ctx.build_reveal_model(
            sender=self.sender,
            model_id=model_id,
            weights_url=weights_url,
            weights_checksum=commitment(encrypted_weights),
            weights_size=len(encrypted_weights),
            decryption_key=decryption_key,
            embedding=embedding,
        )
        await self._ctx.execute(tx_data, "RevealModel")

    # -- Data submission ---------------------------------------------------

    async def submit_data(
        self,
        *,
        target_id: str,
        data: bytes,
        data_url: str,
        model_id: str,
        embedding: list[float],
        distance_score: float,
    ) -> None:
        """Build + execute SubmitData, auto-selecting bond and gas coins.

        Commitment, checksum, and size are computed from *data* automatically.
        """
        data_hash = commitment(data)

        gas_objects = await self._ctx.get_gas_objects(self.sender)
        bond_coin = json.loads(gas_objects[0])
        gas_coin = json.loads(gas_objects[1]) if len(gas_objects) > 1 else None

        tx_data = await self._ctx.build_submit_data(
            sender=self.sender,
            target_id=target_id,
            data_commitment=data_hash,
            data_url=data_url,
            data_checksum=data_hash,
            data_size=len(data),
            model_id=model_id,
            embedding=embedding,
            distance_score=distance_score,
            bond_coin=bond_coin,
            gas=gas_coin,
        )
        await self._ctx.execute(tx_data, "SubmitData")

    # -- Rewards -----------------------------------------------------------

    async def claim_rewards(self, *, target_id: str) -> None:
        """Build + execute ClaimRewards."""
        tx_data = await self._ctx.build_claim_rewards(
            sender=self.sender, target_id=target_id,
        )
        await self._ctx.execute(tx_data, "ClaimRewards")


# ---------------------------------------------------------------------------
# Standalone helpers (work without Wallet)
# ---------------------------------------------------------------------------

async def get_targets(
    client: "SomaClient",
    status: str | None = None,
    epoch: int | None = None,
    limit: int | None = None,
) -> list[Target]:
    """List targets as typed ``Target`` objects (wraps ``client.list_targets()``)."""
    raw = json.loads(await client.list_targets(status=status, epoch=epoch, limit=limit))
    return [Target._from_json(t) for t in raw.get("targets", [])]


# ---------------------------------------------------------------------------
# Localnet helpers
# ---------------------------------------------------------------------------

def advance_epoch(admin_url: str = "http://127.0.0.1:9125") -> int:
    """POST to admin endpoint to trigger epoch advancement. Returns new epoch.

    Only works on localnet with the admin server running
    (started automatically by ``soma start``).
    """
    req = Request(f"{admin_url}/advance-epoch", method="POST", data=b"")
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["epoch"]


async def wait_for_next_epoch(client: "SomaClient", timeout: float = 120.0) -> int:
    """Poll ``get_latest_system_state()`` until epoch changes.

    Fallback for when the admin endpoint isn't available.
    """
    import asyncio
    import time

    state = json.loads(await client.get_latest_system_state())
    start_epoch = state["epoch"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(1.0)
        state = json.loads(await client.get_latest_system_state())
        if state["epoch"] > start_epoch:
            return state["epoch"]
    raise TimeoutError(f"Epoch did not advance within {timeout}s")


def request_faucet(
    address: str,
    faucet_url: str = "http://127.0.0.1:9123",
) -> dict:
    """Request funds from the localnet faucet. Returns the faucet response dict."""
    req = Request(
        f"{faucet_url}/gas",
        data=json.dumps({"FixedAmountRequest": {"recipient": address}}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req) as resp:
        return json.loads(resp.read())


__all__ = [
    "SHANNONS_PER_SOMA",
    "to_shannons",
    "to_soma",
    "SomaClient",
    "WalletContext",
    "Wallet",
    "Target",
    "ScoringClient",
    "ScoreResult",
    "ModelManifest",
    "blake2b_commitment",
    "commitment",
    "encrypt_weights",
    "random_object_id",
    "get_targets",
    "advance_epoch",
    "wait_for_next_epoch",
    "request_faucet",
]
