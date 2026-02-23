# soma-sdk

Python SDK for interacting with the Soma network. Built with [PyO3](https://pyo3.rs) and [Maturin](https://www.maturin.rs), providing native-speed bindings to the Rust SDK.

## Install

```bash
pip install soma-sdk
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add soma-sdk
```

**Requires Python >= 3.10.**

## Quick Start

```python
import asyncio
from soma_sdk import SomaClient, Keypair

async def main():
    # Connect to a local node (started with `soma start localnet`)
    client = await SomaClient(
        "http://localhost:9000",
        faucet_url="http://localhost:9123",
        scoring_url="http://localhost:9124",
    )

    # Generate a keypair and fund it
    keypair = Keypair.generate()
    await client.request_faucet(keypair.address())

    # Check balance (1 SOMA = 1_000_000_000 shannons)
    balance = await client.get_balance(keypair.address())
    print(f"Balance: {SomaClient.to_soma(balance)} SOMA")

    # Query network state
    state = await client.get_latest_system_state()
    print(f"Epoch: {state.epoch}, Validators: {len(state.validators.validators)}")

asyncio.run(main())
```

## Classes

### `Keypair`

Ed25519 keypair for signing Soma transactions.

```python
# Generate a new random keypair
keypair = Keypair.generate()

# Restore from a secret key (hex string or bytes)
keypair = Keypair.from_secret_key("0x...")

# Restore from a BIP-39 mnemonic
keypair = Keypair.from_mnemonic("word1 word2 ...")

keypair.address()        # Soma address (hex string)
keypair.to_secret_key()  # Export secret key (hex string)
keypair.sign(tx_bytes)   # Sign raw transaction data bytes
```

### `SomaClient`

The main client for querying chain state, building transactions, and executing them via gRPC.

```python
client = await SomaClient(
    "http://localhost:9000",
    scoring_url="http://localhost:9124",   # optional — for scoring
    faucet_url="http://localhost:9123",    # optional — for faucet
    admin_url="http://localhost:9125",     # optional — for epoch advancement (localnet)
)
```

#### Chain & Node Info

| Method | Returns | Description |
|--------|---------|-------------|
| `get_chain_identifier()` | `str` | Chain identifier string |
| `get_server_version()` | `str` | Server version string |
| `get_protocol_version()` | `int` | Current protocol version |
| `get_architecture_version()` | `int` | Current model architecture version |
| `get_embedding_dim()` | `int` | Embedding dimension for the current protocol |
| `get_model_min_stake()` | `int` | Minimum stake required to register a model (shannons) |
| `check_api_version()` | `None` | Raises if client/server versions mismatch |

#### Objects & State

| Method | Returns | Description |
|--------|---------|-------------|
| `get_object(object_id)` | `ObjectRef` | Get object by hex ID |
| `get_object_with_version(object_id, version)` | `ObjectRef` | Get object at a specific version |
| `get_balance(address)` | `int` | Balance in shannons |
| `get_latest_system_state()` | `SystemState` | Current global system state |
| `get_epoch(epoch=None)` | `EpochInfo` | Epoch info (`None` for latest) |
| `list_owned_objects(owner, object_type=None, limit=None)` | `list[ObjectRef]` | Objects owned by an address |

`object_type` can be: `"coin"`, `"staked_soma"`, `"target"`, `"submission"`, `"challenge"`, `"system_state"`.

#### Targets & Challenges

| Method | Returns | Description |
|--------|---------|-------------|
| `list_targets(status=None, epoch=None, limit=None)` | `ListTargetsResponse` | List targets with optional filters |
| `get_targets(status=None, epoch=None, limit=None)` | `list[Target]` | Convenience wrapper returning target list directly |
| `get_model_manifests(model_ids_or_target)` | `list[ModelManifest]` | Get model weight manifests by IDs or from a Target |
| `get_challenge(challenge_id)` | `ChallengeInfo` | Get challenge by ID |
| `list_challenges(target_id=None, status=None, ...)` | `ListChallengesResponse` | List challenges with optional filters |

#### Checkpoints

| Method | Returns | Description |
|--------|---------|-------------|
| `get_latest_checkpoint()` | `CheckpointSummary` | Latest checkpoint summary |
| `get_checkpoint_summary(sequence_number)` | `CheckpointSummary` | Checkpoint by sequence number |

#### Transactions

| Method | Returns | Description |
|--------|---------|-------------|
| `execute_transaction(tx_bytes)` | `TransactionEffects` | Execute a signed transaction (BCS bytes) |
| `simulate_transaction(tx_data_bytes)` | `TransactionEffects` | Simulate unsigned transaction data (BCS bytes) |
| `get_transaction(digest)` | `TransactionEffects` | Get transaction effects by digest |

#### Scoring (requires `scoring_url`)

| Method | Returns | Description |
|--------|---------|-------------|
| `score(data_url, models, target_embedding, ...)` | `ScoreResult` | Score data against models |
| `scoring_health()` | `bool` | Check if the scoring service is reachable |

#### Faucet (requires `faucet_url`)

| Method | Returns | Description |
|--------|---------|-------------|
| `request_faucet(address)` | `FaucetResponse` | Request test tokens |

#### Proxy Fetch (via fullnode)

| Method | Returns | Description |
|--------|---------|-------------|
| `fetch_model(model_id)` | `bytes` | Download model weights through the fullnode proxy |
| `fetch_submission_data(target_id)` | `bytes` | Download submission data through the fullnode proxy |

#### Epoch Helpers

| Method | Returns | Description |
|--------|---------|-------------|
| `wait_for_next_epoch(timeout=120.0)` | `int` | Block until the next epoch starts; returns new epoch number |
| `advance_epoch()` | `int` | Force epoch advancement (requires `admin_url`, localnet only) |

#### Static Utilities

```python
SomaClient.to_shannons(1.5)                # 1_500_000_000
SomaClient.to_soma(1_500_000_000)           # 1.5
SomaClient.commitment(data)                 # Blake2b-256 hex digest
SomaClient.encrypt_weights(data)            # (encrypted_bytes, key_hex)
SomaClient.decrypt_weights(data, key_hex)   # decrypted bytes
```

---

## Transaction Builders

All builders are methods on `SomaClient` and return `bytes` (BCS-encoded `TransactionData`). Sign with `Keypair.sign()` and execute with `client.execute_transaction()`.

The `gas` parameter is always optional — when `None`, a gas coin is auto-selected from the sender's owned coins.

### Coin & Object Transfers

```python
# Transfer a coin (optionally a partial amount)
tx = await client.build_transfer_coin(sender, recipient, coin, amount=None, gas=None)

# Transfer arbitrary objects
tx = await client.build_transfer_objects(sender, recipient, [obj1, obj2], gas=None)

# Multi-recipient payment
tx = await client.build_pay_coins(sender, recipients, amounts, coins, gas=None)
```

### Staking

```python
# Stake with a validator
tx = await client.build_add_stake(sender, validator, coin, amount=None, gas=None)

# Withdraw stake
tx = await client.build_withdraw_stake(sender, staked_soma, gas=None)

# Stake with a model
tx = await client.build_add_stake_to_model(sender, model_id, coin, amount=None, gas=None)
```

### Model Management

```python
# Register a model (commit-reveal pattern)
tx = await client.build_commit_model(
    sender, model_id,
    weights_url_commitment,    # Blake2b-256 hex of the weights URL
    weights_commitment,        # Blake2b-256 hex of the encrypted weights
    stake_amount,              # int (shannons)
    commission_rate,           # int (BPS, 10000 = 100%)
    staking_pool_id,           # hex object ID
    gas=None,
)

# Reveal model weights (must be called the epoch after commit)
tx = await client.build_reveal_model(
    sender, model_id,
    weights_url,               # URL string
    weights_checksum,          # Blake2b-256 hex of raw weights
    weights_size,              # int (bytes)
    decryption_key,            # AES-256 hex key
    embedding,                 # list[float] — model embedding vector
    gas=None,
)

# Update model weights (commit-reveal)
tx = await client.build_commit_model_update(sender, model_id, weights_url_commitment, weights_commitment, gas=None)
tx = await client.build_reveal_model_update(sender, model_id, weights_url, weights_checksum, weights_size, decryption_key, embedding, gas=None)

# Other model operations
tx = await client.build_deactivate_model(sender, model_id, gas=None)
tx = await client.build_set_model_commission_rate(sender, model_id, new_rate, gas=None)
tx = await client.build_report_model(sender, model_id, gas=None)
tx = await client.build_undo_report_model(sender, model_id, gas=None)
```

### Data Submissions

```python
# Submit data to fill a target
tx = await client.build_submit_data(
    sender, target_id,
    data_commitment,           # Blake2b-256 hex
    data_url,                  # URL string
    data_checksum,             # Blake2b-256 hex
    data_size,                 # int (bytes)
    model_id,                  # hex object ID
    embedding,                 # list[float]
    distance_score,            # float
    bond_coin,                 # ObjectRef
    gas=None,
)

# Claim rewards from a filled/expired target
tx = await client.build_claim_rewards(sender, target_id, gas=None)

# Report a fraudulent submission
tx = await client.build_report_submission(sender, target_id, challenger=None, gas=None)
tx = await client.build_undo_report_submission(sender, target_id, gas=None)
```

### Challenges

```python
# Initiate a challenge against a filled target
tx = await client.build_initiate_challenge(sender, target_id, bond_coin, gas=None)

# Report/undo challenge
tx = await client.build_report_challenge(sender, challenge_id, gas=None)
tx = await client.build_undo_report_challenge(sender, challenge_id, gas=None)

# Resolve and claim challenge bond
tx = await client.build_claim_challenge_bond(sender, challenge_id, gas=None)
```

### Validator Management

```python
tx = await client.build_add_validator(sender, pubkey_bytes, network_pubkey_bytes, worker_pubkey_bytes, proof_of_possession, net_address, p2p_address, primary_address, proxy_address, gas=None)
tx = await client.build_remove_validator(sender, pubkey_bytes, gas=None)
tx = await client.build_update_validator_metadata(sender, gas=None, next_epoch_network_address=None, ...)
tx = await client.build_set_commission_rate(sender, new_rate, gas=None)
tx = await client.build_report_validator(sender, reportee, gas=None)
tx = await client.build_undo_report_validator(sender, reportee, gas=None)
```

---

## High-Level Convenience Methods

These methods handle building, signing, and executing in one call:

```python
# Register a model (commit step)
model_id = await client.commit_model(
    signer=keypair,
    weights_url="https://storage.example.com/weights.safetensors",
    encrypted_weights=encrypted_bytes,
    commission_rate=1000,          # 10%
    stake_amount=10_000_000_000,   # optional, defaults to minimum
)

# Reveal model weights (next epoch)
await client.reveal_model(
    signer=keypair,
    model_id=model_id,
    weights_url="https://storage.example.com/weights.safetensors",
    encrypted_weights=encrypted_bytes,
    decryption_key=key_hex,
    embedding=[0.1, 0.2, ...],
)

# Submit data to a target
await client.submit_data(
    signer=keypair,
    target_id="0xTARGET",
    data=raw_bytes,
    data_url="https://storage.example.com/data.bin",
    model_id="0xMODEL",
    embedding=[0.1, 0.2, ...],
    distance_score=0.42,
)

# Claim rewards
await client.claim_rewards(signer=keypair, target_id="0xTARGET")
```

---

## `WalletContext`

Manages keys from a local wallet config file (e.g. `~/.soma/client.yaml`). Useful for CLI-style workflows.

```python
wallet = WalletContext("/path/to/client.yaml")
```

| Method | Returns | Description |
|--------|---------|-------------|
| `get_addresses()` | `list[str]` | All managed addresses |
| `active_address()` | `str` | Currently active address |
| `has_addresses()` | `bool` | Whether any addresses exist |
| `get_gas_objects(address)` | `list[ObjectRef]` | Gas coin objects for an address |
| `save_config()` | `None` | Persist wallet config to disk |
| `sign_transaction(tx_data_bytes)` | `bytes` | Sign BCS `TransactionData`, returns BCS `Transaction` |
| `sign_and_execute_transaction(tx_data_bytes)` | `TransactionEffects` | Sign, execute, and wait for checkpoint inclusion |
| `sign_and_execute_transaction_may_fail(tx_data_bytes)` | `TransactionEffects` | Same as above but returns effects even on failure |

## Building from Source

Requires Rust and Python >= 3.10.

```bash
# Install maturin
pip install maturin

# Development build (editable install)
cd python-sdk
maturin develop

# Release build
maturin build --release
```

## License

Apache-2.0
