# soma-sdk

Python SDK for interacting with the SOMA network. Built with [PyO3](https://pyo3.rs) and [Maturin](https://www.maturin.rs), providing native-speed bindings to the Rust SDK.

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
    # Connect using a chain preset
    client = await SomaClient(chain="testnet")   # or chain="localnet"

    # Generate a keypair and fund it
    keypair = Keypair.generate()
    await client.request_faucet(keypair.address())

    # Check balance (returned as SOMA float)
    balance = await client.get_balance(keypair.address())
    print(f"Balance: {balance:.2f} SOMA")

    # Query network state
    state = await client.get_latest_system_state()
    print(f"Epoch: {state.epoch}, Validators: {len(state.validators.validators)}")

asyncio.run(main())
```

## Classes

### `Keypair`

Ed25519 keypair for signing SOMA transactions.

```python
# Generate a new random keypair
keypair = Keypair.generate()

# Restore from a secret key (hex string or bytes)
keypair = Keypair.from_secret_key("0x...")

# Restore from a BIP-39 mnemonic
keypair = Keypair.from_mnemonic("word1 word2 ...")

keypair.address()        # SOMA address (hex string — addresses use hex)
keypair.to_secret_key()  # Export secret key (hex string)
keypair.sign(tx_bytes)   # Sign raw transaction data bytes
```

### `SomaClient`

The main client for querying chain state, building transactions, and executing them via gRPC.

```python
# Named chain presets (recommended)
client = await SomaClient(chain="testnet")    # auto-configures RPC + faucet URLs
client = await SomaClient(chain="localnet")   # auto-configures RPC + faucet + admin + scoring

# Explicit URLs
client = await SomaClient("https://fullnode.testnet.soma.org", faucet_url="https://faucet.testnet.soma.org")

# Override scoring_url with any chain preset
client = await SomaClient(chain="testnet", scoring_url="http://my-scorer:9124")
```

Chain presets:
- `"testnet"` — `https://fullnode.testnet.soma.org` + faucet
- `"localnet"` — `http://127.0.0.1:9000` + faucet, admin, scoring

> **Note:** `chain` and `rpc_url`/`faucet_url` cannot be combined (raises `ValueError`). `scoring_url` is always configurable.

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
| `get_balance(address)` | `float` | Balance in SOMA |
| `get_latest_system_state()` | `SystemState` | Current global system state |
| `get_epoch(epoch=None)` | `EpochInfo` | Epoch info (`None` for latest) |
| `list_owned_objects(owner, object_type=None, limit=None)` | `list[ObjectRef]` | Objects owned by an address |

`object_type` can be: `"coin"`, `"staked_soma"`, `"target"`, `"submission"`, `"system_state"`.

#### Targets

| Method | Returns | Description |
|--------|---------|-------------|
| `list_targets(status=None, epoch=None, limit=None)` | `ListTargetsResponse` | List targets with optional filters |
| `get_targets(status=None, epoch=None, limit=None)` | `list[Target]` | Convenience wrapper returning target list directly |
| `get_model_manifests(model_ids_or_target)` | `list[ModelManifest]` | Get model weight manifests by IDs or from a Target |

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
SomaClient.commitment(data)                 # Blake2b-256 Base58 digest
SomaClient.encrypt_weights(data)            # (encrypted_bytes, key_base58)
SomaClient.decrypt_weights(data, key)       # decrypted bytes (key: bytes or Base58 string)
```

---

## High-Level Convenience Methods

These methods handle building, signing, and executing in one call. Amounts are in SOMA (float), not shannons.

```python
# Transfer coins
await client.transfer_coin(signer=keypair, recipient="0xADDR", amount=0.5)

# Multi-recipient payment
await client.pay_coins(signer=keypair, recipients=["0xA", "0xB"], amounts=[0.25, 0.25])

# Stake with a validator
await client.add_stake(signer=keypair, validator="0xVALIDATOR", amount=10.0)

# Register a model (commit step — publishes manifest + commitments)
await client.commit_model(
    signer=keypair,
    weights_url="https://storage.example.com/weights.safetensors",
    encrypted_weights=encrypted_bytes,
    decryption_key=key_base58,
    embedding=[0.1, 0.2, ...],
    commission_rate=1000,          # 10%
    stake_amount=10.0,             # SOMA — optional, defaults to minimum
)

# Reveal model (next epoch — provides decryption key + embedding)
await client.reveal_model(
    signer=keypair,
    model_id=model_id,
    decryption_key=key_base58,
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
    loss_score=[0.1, 0.2, ...],
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
