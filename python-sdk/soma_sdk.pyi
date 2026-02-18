# soma_sdk.pyi
from collections.abc import Generator
from typing import Optional

class SomaClient:
    def __init__(self, rpc_url: str) -> None: ...
    def __await__(self) -> Generator[None, None, "SomaClient"]: ...

    # Chain & Node Info
    async def get_chain_identifier(self) -> str: ...
    async def get_server_version(self) -> str: ...
    async def get_protocol_version(self) -> int: ...
    async def check_api_version(self) -> None: ...

    # Objects & State
    async def get_object(self, object_id: str) -> str: ...
    async def get_object_with_version(self, object_id: str, version: int) -> str: ...
    async def get_balance(self, address: str) -> int: ...
    async def get_latest_system_state(self) -> str: ...
    async def get_epoch(self, epoch: Optional[int] = None) -> str: ...
    async def list_owned_objects(
        self,
        owner: str,
        object_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[str]: ...

    # Targets & Challenges
    async def list_targets(
        self,
        status: Optional[str] = None,
        epoch: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> str: ...
    async def get_challenge(self, challenge_id: str) -> str: ...
    async def list_challenges(
        self,
        target_id: Optional[str] = None,
        status: Optional[str] = None,
        epoch: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> str: ...

    # Checkpoints
    async def get_latest_checkpoint(self) -> str: ...
    async def get_checkpoint_summary(self, sequence_number: int) -> str: ...

    # Transactions
    async def execute_transaction(self, tx_bytes: bytes) -> str: ...
    async def simulate_transaction(self, tx_data_bytes: bytes) -> str: ...
    async def get_transaction(self, digest: str) -> str: ...

class WalletContext:
    def __init__(self, config_path: str) -> None: ...

    # Key Management
    async def get_addresses(self) -> list[str]: ...
    async def active_address(self) -> str: ...
    async def has_addresses(self) -> bool: ...
    async def get_gas_objects(self, address: str) -> list[str]: ...
    async def save_config(self) -> None: ...

    # Signing & Execution
    async def sign_transaction(self, tx_data_bytes: bytes) -> bytes: ...
    async def sign_and_execute_transaction(self, tx_data_bytes: bytes) -> str: ...
    async def sign_and_execute_transaction_may_fail(self, tx_data_bytes: bytes) -> str: ...

    # Transaction Builders — Coin & Object
    async def build_transfer_coin(
        self,
        sender: str,
        recipient: str,
        coin: dict,
        amount: Optional[int] = None,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_transfer_objects(
        self,
        sender: str,
        recipient: str,
        objects: list[dict],
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_pay_coins(
        self,
        sender: str,
        recipients: list[str],
        amounts: list[int],
        coins: list[dict],
        gas: Optional[dict] = None,
    ) -> bytes: ...

    # Transaction Builders — Staking
    async def build_add_stake(
        self,
        sender: str,
        validator: str,
        coin: dict,
        amount: Optional[int] = None,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_withdraw_stake(
        self,
        sender: str,
        staked_soma: dict,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_add_stake_to_model(
        self,
        sender: str,
        model_id: str,
        coin: dict,
        amount: Optional[int] = None,
        gas: Optional[dict] = None,
    ) -> bytes: ...

    # Transaction Builders — Model Management
    async def build_commit_model(
        self,
        sender: str,
        model_id: str,
        weights_url_commitment: str,
        weights_commitment: str,
        architecture_version: int,
        stake_amount: int,
        commission_rate: int,
        staking_pool_id: str,
        gas: Optional[dict] = None,
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
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_commit_model_update(
        self,
        sender: str,
        model_id: str,
        weights_url_commitment: str,
        weights_commitment: str,
        gas: Optional[dict] = None,
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
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_deactivate_model(
        self,
        sender: str,
        model_id: str,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_set_model_commission_rate(
        self,
        sender: str,
        model_id: str,
        new_rate: int,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_report_model(
        self,
        sender: str,
        model_id: str,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_undo_report_model(
        self,
        sender: str,
        model_id: str,
        gas: Optional[dict] = None,
    ) -> bytes: ...

    # Transaction Builders — Submissions
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
        bond_coin: dict,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_claim_rewards(
        self,
        sender: str,
        target_id: str,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_report_submission(
        self,
        sender: str,
        target_id: str,
        challenger: Optional[str] = None,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_undo_report_submission(
        self,
        sender: str,
        target_id: str,
        gas: Optional[dict] = None,
    ) -> bytes: ...

    # Transaction Builders — Challenges
    async def build_initiate_challenge(
        self,
        sender: str,
        target_id: str,
        bond_coin: dict,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_report_challenge(
        self,
        sender: str,
        challenge_id: str,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_undo_report_challenge(
        self,
        sender: str,
        challenge_id: str,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_claim_challenge_bond(
        self,
        sender: str,
        challenge_id: str,
        gas: Optional[dict] = None,
    ) -> bytes: ...

    # Transaction Builders — Validator Management
    async def build_add_validator(
        self,
        sender: str,
        pubkey_bytes: bytes,
        network_pubkey_bytes: bytes,
        worker_pubkey_bytes: bytes,
        net_address: bytes,
        p2p_address: bytes,
        primary_address: bytes,
        proxy_address: bytes,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_remove_validator(
        self,
        sender: str,
        pubkey_bytes: bytes,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_update_validator_metadata(
        self,
        sender: str,
        gas: Optional[dict] = None,
        next_epoch_network_address: Optional[bytes] = None,
        next_epoch_p2p_address: Optional[bytes] = None,
        next_epoch_primary_address: Optional[bytes] = None,
        next_epoch_proxy_address: Optional[bytes] = None,
        next_epoch_protocol_pubkey: Optional[bytes] = None,
        next_epoch_worker_pubkey: Optional[bytes] = None,
        next_epoch_network_pubkey: Optional[bytes] = None,
    ) -> bytes: ...
    async def build_set_commission_rate(
        self,
        sender: str,
        new_rate: int,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_report_validator(
        self,
        sender: str,
        reportee: str,
        gas: Optional[dict] = None,
    ) -> bytes: ...
    async def build_undo_report_validator(
        self,
        sender: str,
        reportee: str,
        gas: Optional[dict] = None,
    ) -> bytes: ...
