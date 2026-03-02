# soma_sdk.pyi — type stubs for IDE autocomplete
from collections.abc import Generator
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Return-type stubs — these describe the SimpleNamespace shapes returned
# by SomaClient methods. They are not importable classes at runtime; they
# exist only for IDE autocomplete and type checking.
# ---------------------------------------------------------------------------

class Target:
    """Shape returned by ``SomaClient.get_targets()`` items."""
    id: str
    status: str
    embedding: list[float]
    model_ids: list[str]
    distance_threshold: float
    reward_pool: int
    generation_epoch: int
    bond_amount: int
    submitter: Optional[str]
    winning_model_id: Optional[str]

class ModelManifest:
    """Shape returned by ``SomaClient.get_model_manifests()`` items."""
    url: str
    checksum: str
    size: int
    decryption_key: Optional[str]

class ScoreResult:
    """Shape returned by ``SomaClient.score()``."""
    winner: int
    loss_score: list[float]
    embedding: list[float]
    distance: list[float]

class FaucetCoinInfo:
    """Shape for individual coin info in faucet response."""
    amount: int
    id: str
    transfer_tx_digest: str

class FaucetResponse:
    """Shape returned by ``SomaClient.request_faucet()``."""
    status: str
    coins_sent: list[FaucetCoinInfo]

class ObjectRef:
    """Shape for object references (e.g. from ``list_owned_objects()``)."""
    id: str
    version: int
    digest: str

class EpochInfo:
    """Shape returned by ``SomaClient.get_epoch()``."""
    epoch: int
    first_checkpoint_id: int
    epoch_start_timestamp_ms: int
    end_of_epoch_info: Optional["EndOfEpochInfo"]

class EndOfEpochInfo:
    """Shape for end-of-epoch info within EpochInfo."""
    last_checkpoint_id: int
    epoch_end_timestamp_ms: Optional[int]

class TransactionEffects:
    """Shape returned by ``execute_transaction()``, ``simulate_transaction()``, ``get_transaction()``."""
    status: str
    gas_used: "GasUsed"
    transaction_digest: str
    created: list["OwnedObjectRef"]
    mutated: list["OwnedObjectRef"]
    deleted: list[str]

class GasUsed:
    """Gas usage details within TransactionEffects."""
    computation_cost: int
    storage_cost: int
    storage_rebate: int
    non_refundable_storage_fee: int

class OwnedObjectRef:
    """Object reference with owner info within TransactionEffects."""
    id: str
    version: int
    digest: str
    owner: str

class PoolTokenExchangeRate:
    """Exchange rate between SOMA and pool tokens at a given epoch."""
    soma_amount: int
    pool_token_amount: int

class StakingPool:
    """Staking pool for a validator."""
    id: str
    activation_epoch: Optional[int]
    deactivation_epoch: Optional[int]
    soma_balance: int
    rewards_pool: int
    pool_token_balance: int
    exchange_rates: dict[int, PoolTokenExchangeRate]
    pending_stake: int
    pending_total_soma_withdraw: int
    pending_pool_token_withdraw: int

class ValidatorMetadata:
    """Metadata for a validator."""
    soma_address: str
    protocol_pubkey: str
    network_pubkey: str
    worker_pubkey: str
    net_address: str
    p2p_address: str
    primary_address: str
    proxy_address: str
    proof_of_possession: str
    next_epoch_protocol_pubkey: Optional[str]
    next_epoch_network_pubkey: Optional[str]
    next_epoch_net_address: Optional[str]
    next_epoch_p2p_address: Optional[str]
    next_epoch_primary_address: Optional[str]
    next_epoch_worker_pubkey: Optional[str]
    next_epoch_proof_of_possession: Optional[str]
    next_epoch_proxy_address: Optional[str]

class Validator:
    """A validator in the validator set."""
    metadata: ValidatorMetadata
    voting_power: int
    staking_pool: StakingPool
    commission_rate: int
    next_epoch_stake: int
    next_epoch_commission_rate: int

class ValidatorSet:
    """The full validator set."""
    total_stake: int
    validators: list[Validator]
    pending_validators: list[Validator]
    pending_removals: list[int]
    staking_pool_mappings: dict[str, str]
    inactive_validators: dict[str, Validator]
    at_risk_validators: dict[str, int]

class SystemParameters:
    """On-chain system parameters (protocol config)."""
    epoch_duration_ms: int
    validator_reward_allocation_bps: int
    model_min_stake: int
    model_architecture_version: int
    model_reveal_slash_rate_bps: int
    model_tally_slash_rate_bps: int
    target_epoch_fee_collection: int
    base_fee: int
    write_object_fee: int
    value_fee_bps: int
    min_value_fee_bps: int
    max_value_fee_bps: int
    fee_adjustment_rate_bps: int
    target_models_per_target: int
    target_embedding_dim: int
    target_initial_distance_threshold: object
    target_reward_allocation_bps: int
    target_hits_per_epoch: int
    target_hits_ema_decay_bps: int
    target_difficulty_adjustment_rate_bps: int
    target_max_distance_threshold: object
    target_min_distance_threshold: object
    target_initial_targets_per_epoch: int
    target_submitter_reward_share_bps: int
    target_model_reward_share_bps: int
    target_claimer_incentive_bps: int
    submission_bond_per_byte: int
    max_submission_data_size: int

class EmissionPool:
    """Emission pool state."""
    balance: int
    emission_per_epoch: int

class TargetState:
    """Target generation state."""
    distance_threshold: object
    targets_generated_this_epoch: int
    hits_this_epoch: int
    hits_ema: int
    reward_per_target: int

class SystemState:
    """Shape returned by ``SomaClient.get_latest_system_state()``."""
    epoch: int
    protocol_version: int
    validators: ValidatorSet
    parameters: SystemParameters
    epoch_start_timestamp_ms: int
    validator_report_records: dict[str, list[str]]
    model_registry: object
    emission_pool: EmissionPool
    target_state: TargetState
    safe_mode: bool
    safe_mode_accumulated_fees: int
    safe_mode_accumulated_emissions: int

class ListTargetsResponse:
    """Shape returned by ``SomaClient.list_targets()``."""
    targets: list[Target]
    next_page_token: Optional[str]

class CheckpointSummary:
    """Shape returned by checkpoint methods."""
    class CheckpointData:
        sequence_number: int
        epoch: int
        network_total_transactions: int
        content_digest: str
        timestamp_ms: int
    data: CheckpointData

# ---------------------------------------------------------------------------
# Keypair
# ---------------------------------------------------------------------------

class Keypair:
    """Ed25519 keypair for signing SOMA transactions."""
    @staticmethod
    def generate() -> "Keypair": ...
    @staticmethod
    def from_secret_key(secret: Union[bytes, str]) -> "Keypair": ...
    @staticmethod
    def from_mnemonic(mnemonic: str) -> "Keypair": ...
    def address(self) -> str: ...
    def sign(self, tx_data_bytes: bytes) -> bytes: ...
    def to_secret_key(self) -> str: ...

# ---------------------------------------------------------------------------
# SomaClient
# ---------------------------------------------------------------------------

class SomaClient:
    """A client for interacting with the SOMA network via gRPC.

    Use ``chain`` for named network presets::

        client = await SomaClient(chain="testnet")
        client = await SomaClient(chain="localnet")

    Or provide an explicit ``rpc_url``::

        client = await SomaClient("https://fullnode.testnet.soma.org")
    """
    def __init__(
        self,
        rpc_url: Optional[str] = None,
        *,
        chain: Optional[str] = None,
        scoring_url: Optional[str] = None,
        admin_url: Optional[str] = None,
        faucet_url: Optional[str] = None,
    ) -> None: ...
    def __await__(self) -> Generator[None, None, "SomaClient"]: ...

    # -- Static crypto utility methods --
    @staticmethod
    def encrypt_weights(
        data: bytes,
        key: Optional[bytes] = None,
    ) -> tuple[bytes, str]:
        """Encrypt with AES-256-CTR (zero IV). Returns (encrypted_bytes, key_hex)."""
        ...
    @staticmethod
    def decrypt_weights(
        data: bytes,
        key: Union[bytes, str],
    ) -> bytes:
        """Decrypt with AES-256-CTR (zero IV). Key can be bytes or hex string."""
        ...
    @staticmethod
    def commitment(data: bytes) -> str:
        """Blake2b-256 hash of data, returned as 64-char hex string."""
        ...
    @staticmethod
    def to_shannons(soma: float) -> int:
        """Convert SOMA to shannons (1 SOMA = 1,000,000,000 shannons)."""
        ...
    @staticmethod
    def to_soma(shannons: int) -> float:
        """Convert shannons to SOMA."""
        ...

    # -- Chain & Node Info --
    async def get_chain_identifier(self) -> str: ...
    async def get_server_version(self) -> str: ...
    async def get_protocol_version(self) -> int: ...
    async def get_architecture_version(self) -> int: ...
    async def get_embedding_dim(self) -> int: ...
    async def get_model_min_stake(self) -> int: ...
    async def check_api_version(self) -> None: ...

    # -- Objects & State --
    async def get_object(self, object_id: str) -> ObjectRef: ...
    async def get_object_with_version(self, object_id: str, version: int) -> ObjectRef: ...
    async def get_balance(self, address: str) -> float: ...
    async def get_latest_system_state(self) -> SystemState: ...
    async def get_epoch(self, epoch: Optional[int] = None) -> EpochInfo: ...
    async def list_owned_objects(
        self,
        owner: str,
        object_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[ObjectRef]: ...

    # -- Targets --
    async def list_targets(
        self,
        status: Optional[str] = None,
        epoch: Optional[int] = None,
        limit: Optional[int] = None,
        read_mask: Optional[str] = None,
    ) -> ListTargetsResponse: ...
    async def get_targets(
        self,
        status: Optional[str] = None,
        epoch: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[Target]: ...
    async def get_model_manifests(
        self,
        model_ids_or_target: Union[list[str], Target],
    ) -> list[ModelManifest]: ...

    # -- Checkpoints --
    async def get_latest_checkpoint(self) -> CheckpointSummary: ...
    async def get_checkpoint_summary(self, sequence_number: int) -> CheckpointSummary: ...

    # -- Transactions --
    async def execute_transaction(self, tx_bytes: bytes) -> TransactionEffects: ...
    async def simulate_transaction(self, tx_data_bytes: bytes) -> TransactionEffects: ...
    async def get_transaction(self, digest: str) -> TransactionEffects: ...

    # -- Scoring (requires scoring_url) --
    async def score(
        self,
        data_url: str,
        models: list[ModelManifest],
        target_embedding: list[float],
        data: Optional[bytes] = None,
        data_checksum: Optional[str] = None,
        data_size: Optional[int] = None,
        seed: int = 0,
    ) -> ScoreResult: ...
    async def scoring_health(self) -> bool: ...

    # -- Admin (requires admin_url) --
    async def advance_epoch(self) -> int: ...

    # -- Faucet (requires faucet_url) --
    async def request_faucet(self, address: str) -> FaucetResponse: ...

    # -- Proxy Client (fetch via fullnode proxy) --
    async def fetch_model(self, model_id: str) -> bytes: ...
    async def fetch_submission_data(self, target_id: str) -> bytes: ...

    # -- Epoch Helpers --
    async def wait_for_next_epoch(self, timeout: float = 120.0) -> int: ...
    async def get_next_epoch_timestamp(self) -> int:
        """Return estimated UNIX timestamp (ms) when the next epoch starts.

        Computed as: epoch_start_timestamp_ms + epoch_duration_ms from the
        current system state. Useful for cron scheduling.
        """
        ...
    async def get_following_epoch_timestamp(self) -> int:
        """Return estimated UNIX timestamp (ms) when the epoch after next starts.

        Computed as: epoch_start_timestamp_ms + 2 * epoch_duration_ms.
        Useful for scheduling actions that require two epoch boundaries
        (e.g., claim_rewards after submit_data).
        """
        ...

    # -- Transaction Builders: Coin & Object --
    async def build_transfer_coin(
        self,
        sender: str,
        recipient: str,
        coin: ObjectRef,
        amount: Optional[int] = None,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_transfer_objects(
        self,
        sender: str,
        recipient: str,
        objects: list[ObjectRef],
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_pay_coins(
        self,
        sender: str,
        recipients: list[str],
        amounts: list[int],
        coins: list[ObjectRef],
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...

    # -- Transaction Builders: Staking --
    async def build_add_stake(
        self,
        sender: str,
        validator: str,
        coin: ObjectRef,
        amount: Optional[int] = None,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_withdraw_stake(
        self,
        sender: str,
        staked_soma: ObjectRef,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_add_stake_to_model(
        self,
        sender: str,
        model_id: str,
        coin: ObjectRef,
        amount: Optional[int] = None,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...

    # -- Transaction Builders: Model Management --
    async def build_commit_model(
        self,
        sender: str,
        model_id: str,
        weights_url_commitment: str,
        weights_commitment: str,
        stake_amount: int,
        commission_rate: int,
        staking_pool_id: str,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_reveal_model(
        self,
        sender: str,
        model_id: str,
        weights_url: str,
        weights_checksum: str,
        weights_size: int,
        decryption_key: str,
        embedding: list[float],
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_commit_model_update(
        self,
        sender: str,
        model_id: str,
        weights_url_commitment: str,
        weights_commitment: str,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_reveal_model_update(
        self,
        sender: str,
        model_id: str,
        weights_url: str,
        weights_checksum: str,
        weights_size: int,
        decryption_key: str,
        embedding: list[float],
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_deactivate_model(
        self,
        sender: str,
        model_id: str,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_set_model_commission_rate(
        self,
        sender: str,
        model_id: str,
        new_rate: int,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_report_model(
        self,
        sender: str,
        model_id: str,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_undo_report_model(
        self,
        sender: str,
        model_id: str,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...

    # -- Transaction Builders: Submissions --
    async def build_submit_data(
        self,
        sender: str,
        target_id: str,
        data_commitment: str,
        data_url: str,
        data_checksum: str,
        data_size: int,
        model_id: str,
        embedding: list[float],
        distance_score: float,
        bond_coin: ObjectRef,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_claim_rewards(
        self,
        sender: str,
        target_id: str,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_report_submission(
        self,
        sender: str,
        target_id: str,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_undo_report_submission(
        self,
        sender: str,
        target_id: str,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...

    # -- Transaction Builders: Validator Management --
    async def build_add_validator(
        self,
        sender: str,
        pubkey_bytes: bytes,
        network_pubkey_bytes: bytes,
        worker_pubkey_bytes: bytes,
        proof_of_possession: bytes,
        net_address: bytes,
        p2p_address: bytes,
        primary_address: bytes,
        proxy_address: bytes,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_remove_validator(
        self,
        sender: str,
        pubkey_bytes: bytes,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_update_validator_metadata(
        self,
        sender: str,
        gas: Optional[ObjectRef] = None,
        next_epoch_network_address: Optional[bytes] = None,
        next_epoch_p2p_address: Optional[bytes] = None,
        next_epoch_primary_address: Optional[bytes] = None,
        next_epoch_proxy_address: Optional[bytes] = None,
        next_epoch_protocol_pubkey: Optional[bytes] = None,
        next_epoch_worker_pubkey: Optional[bytes] = None,
        next_epoch_network_pubkey: Optional[bytes] = None,
        next_epoch_proof_of_possession: Optional[bytes] = None,
    ) -> bytes: ...
    async def build_set_commission_rate(
        self,
        sender: str,
        new_rate: int,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_report_validator(
        self,
        sender: str,
        reportee: str,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...
    async def build_undo_report_validator(
        self,
        sender: str,
        reportee: str,
        gas: Optional[ObjectRef] = None,
    ) -> bytes: ...

    # -- High-level convenience methods (build + sign + execute) --
    async def commit_model(
        self,
        signer: Keypair,
        weights_url: str,
        encrypted_weights: bytes,
        commission_rate: int,
        stake_amount: Optional[float] = None,
    ) -> str: ...
    async def reveal_model(
        self,
        signer: Keypair,
        model_id: str,
        weights_url: str,
        encrypted_weights: bytes,
        decryption_key: str,
        embedding: list[float],
    ) -> None: ...
    async def submit_data(
        self,
        signer: Keypair,
        target_id: str,
        data: bytes,
        data_url: str,
        model_id: str,
        embedding: list[float],
        distance_score: float,
    ) -> None: ...
    async def claim_rewards(
        self,
        signer: Keypair,
        target_id: str,
    ) -> None: ...

    # -- High-level: Coin & Object --
    async def transfer_coin(
        self,
        signer: Keypair,
        recipient: str,
        amount: float,
    ) -> None: ...
    async def transfer_objects(
        self,
        signer: Keypair,
        recipient: str,
        object_ids: list[str],
    ) -> None: ...
    async def pay_coins(
        self,
        signer: Keypair,
        recipients: list[str],
        amounts: list[float],
    ) -> None: ...

    # -- High-level: Staking --
    async def add_stake(
        self,
        signer: Keypair,
        validator: str,
        amount: Optional[float] = None,
    ) -> None: ...
    async def withdraw_stake(
        self,
        signer: Keypair,
        staked_soma_id: str,
    ) -> None: ...
    async def add_stake_to_model(
        self,
        signer: Keypair,
        model_id: str,
        amount: Optional[float] = None,
    ) -> None: ...

    # -- High-level: Model Management --
    async def commit_model_update(
        self,
        signer: Keypair,
        model_id: str,
        weights_url: str,
        encrypted_weights: bytes,
    ) -> None: ...
    async def reveal_model_update(
        self,
        signer: Keypair,
        model_id: str,
        weights_url: str,
        encrypted_weights: bytes,
        decryption_key: str,
        embedding: list[float],
    ) -> None: ...
    async def deactivate_model(
        self,
        signer: Keypair,
        model_id: str,
    ) -> None: ...
    async def set_model_commission_rate(
        self,
        signer: Keypair,
        model_id: str,
        new_rate: int,
    ) -> None: ...
    async def report_model(
        self,
        signer: Keypair,
        model_id: str,
    ) -> None: ...
    async def undo_report_model(
        self,
        signer: Keypair,
        model_id: str,
    ) -> None: ...

    # -- High-level: Submission --
    async def report_submission(
        self,
        signer: Keypair,
        target_id: str,
    ) -> None: ...
    async def undo_report_submission(
        self,
        signer: Keypair,
        target_id: str,
    ) -> None: ...

    # -- High-level: Validator Management --
    async def add_validator(
        self,
        signer: Keypair,
        pubkey_bytes: bytes,
        network_pubkey_bytes: bytes,
        worker_pubkey_bytes: bytes,
        proof_of_possession: bytes,
        net_address: bytes,
        p2p_address: bytes,
        primary_address: bytes,
        proxy_address: bytes,
    ) -> None: ...
    async def remove_validator(
        self,
        signer: Keypair,
        pubkey_bytes: bytes,
    ) -> None: ...
    async def update_validator_metadata(
        self,
        signer: Keypair,
        next_epoch_network_address: Optional[bytes] = None,
        next_epoch_p2p_address: Optional[bytes] = None,
        next_epoch_primary_address: Optional[bytes] = None,
        next_epoch_proxy_address: Optional[bytes] = None,
        next_epoch_protocol_pubkey: Optional[bytes] = None,
        next_epoch_worker_pubkey: Optional[bytes] = None,
        next_epoch_network_pubkey: Optional[bytes] = None,
        next_epoch_proof_of_possession: Optional[bytes] = None,
    ) -> None: ...
    async def set_validator_commission_rate(
        self,
        signer: Keypair,
        new_rate: int,
    ) -> None: ...
    async def report_validator(
        self,
        signer: Keypair,
        reportee: str,
    ) -> None: ...
    async def undo_report_validator(
        self,
        signer: Keypair,
        reportee: str,
    ) -> None: ...
