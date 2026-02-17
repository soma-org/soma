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

**Requires Python ≥ 3.10.**

## Quick Start

```python
import asyncio
from soma_sdk import SomaClient, WalletContext

async def main():
    # Connect to a Soma node
    client = await SomaClient("http://localhost:9000")

    # Query chain info
    chain_id = await client.get_chain_identifier()
    version = await client.get_server_version()
    print(f"Chain: {chain_id}, Version: {version}")

    # Check a balance (returns shannons; 1 SOMA = 1_000_000_000 shannons)
    balance = await client.get_balance("0xADDRESS")
    print(f"Balance: {balance} shannons")

asyncio.run(main())
```

## Classes

### `SomaClient`

Read-only client for querying chain state and submitting pre-signed transactions via gRPC.

```python
client = await SomaClient("http://localhost:9000")
```

#### Chain & Node Info

| Method | Returns | Description |
|--------|---------|-------------|
| `get_chain_identifier()` | `str` | Chain identifier string |
| `get_server_version()` | `str` | Server version string |
| `get_protocol_version()` | `int` | Current protocol version |
| `check_api_version()` | `None` | Raises if client/server versions mismatch |

#### Objects & State

| Method | Returns | Description |
|--------|---------|-------------|
| `get_object(object_id)` | `str` (JSON) | Get object by hex ID |
| `get_object_with_version(object_id, version)` | `str` (JSON) | Get object at a specific version |
| `get_balance(address)` | `int` | Balance in shannons |
| `get_latest_system_state()` | `str` (JSON) | Current global system state |
| `get_epoch(epoch=None)` | `str` (JSON) | Epoch info (`None` for latest) |
| `list_owned_objects(owner, object_type=None, limit=None)` | `list[str]` (JSON) | Objects owned by an address |

`object_type` can be: `"coin"`, `"staked_soma"`, `"target"`, `"submission"`, `"challenge"`, `"system_state"`.

#### Targets & Challenges

| Method | Returns | Description |
|--------|---------|-------------|
| `list_targets(status=None, epoch=None, limit=None)` | `str` (JSON) | List targets with optional filters |
| `get_challenge(challenge_id)` | `str` (JSON) | Get challenge by ID |
| `list_challenges(target_id=None, status=None, epoch=None, limit=None)` | `str` (JSON) | List challenges with optional filters |

#### Checkpoints

| Method | Returns | Description |
|--------|---------|-------------|
| `get_latest_checkpoint()` | `str` (JSON) | Latest checkpoint summary |
| `get_checkpoint_summary(sequence_number)` | `str` (JSON) | Checkpoint by sequence number |

#### Transactions

| Method | Returns | Description |
|--------|---------|-------------|
| `execute_transaction(tx_bytes)` | `str` (JSON) | Execute a signed transaction (BCS bytes) |
| `simulate_transaction(tx_data_bytes)` | `str` (JSON) | Simulate unsigned transaction data (BCS bytes) |
| `get_transaction(digest)` | `str` (JSON) | Get transaction effects by digest |

---

### `WalletContext`

Manages keys, builds transactions, signs, and executes. Wraps a local wallet config file (e.g. `~/.soma/client.yaml`).

```python
wallet = WalletContext("/path/to/client.yaml")
```

#### Key Management

| Method | Returns | Description |
|--------|---------|-------------|
| `get_addresses()` | `list[str]` | All managed addresses |
| `active_address()` | `str` | Currently active address |
| `has_addresses()` | `bool` | Whether any addresses exist |
| `get_gas_objects(address)` | `list[str]` (JSON) | Gas coin objects for an address |
| `save_config()` | `None` | Persist wallet config to disk |

#### Signing & Execution

| Method | Returns | Description |
|--------|---------|-------------|
| `sign_transaction(tx_data_bytes)` | `bytes` | Sign BCS `TransactionData`, returns BCS `Transaction` |
| `sign_and_execute_transaction(tx_data_bytes)` | `str` (JSON) | Sign, execute, and wait for checkpoint inclusion. **Panics on failure.** |
| `sign_and_execute_transaction_may_fail(tx_data_bytes)` | `str` (JSON) | Same as above but returns effects even on failure |

#### Transaction Builders

All builders return `bytes` (BCS-encoded `TransactionData`). Pass the result to `sign_transaction` or `sign_and_execute_transaction`.

The `gas` parameter is always optional — when `None`, a gas coin is auto-selected from the sender's owned coins. When provided, it must be a dict with `{"id": str, "version": int, "digest": str}`.

**Coin & Object Transfers**

```python
# Transfer a coin (optionally a partial amount)
tx = await wallet.build_transfer_coin(sender, recipient, coin, amount=None, gas=None)

# Transfer arbitrary objects
tx = await wallet.build_transfer_objects(sender, recipient, [obj1, obj2], gas=None)

# Multi-recipient payment
tx = await wallet.build_pay_coins(sender, recipients, amounts, coins, gas=None)
```

**Staking**

```python
# Stake with a validator
tx = await wallet.build_add_stake(sender, validator, coin, amount=None, gas=None)

# Withdraw stake
tx = await wallet.build_withdraw_stake(sender, staked_soma, gas=None)

# Stake with a model
tx = await wallet.build_add_stake_to_model(sender, model_id, coin, amount=None, gas=None)
```

**Model Management**

```python
# Register a model (commit-reveal pattern)
tx = await wallet.build_commit_model(
    sender, model_id,
    weights_url_commitment,    # 32-byte hex
    weights_commitment,        # 32-byte hex
    architecture_version,      # int
    stake_amount,              # int (shannons)
    commission_rate,           # int (BPS, 10000 = 100%)
    staking_pool_id,           # hex object ID
    gas=None,
)

# Reveal model weights (must be called the epoch after commit)
tx = await wallet.build_reveal_model(
    sender, model_id,
    weights_url,               # URL string
    weights_checksum,          # 32-byte hex
    weights_size,              # int (bytes)
    decryption_key,            # 32-byte hex
    embedding,                 # list[float] — model embedding vector
    gas=None,
)

# Update model weights (commit-reveal)
tx = await wallet.build_commit_model_update(sender, model_id, weights_url_commitment, weights_commitment, gas=None)
tx = await wallet.build_reveal_model_update(sender, model_id, weights_url, weights_checksum, weights_size, decryption_key, embedding, gas=None)

# Other model operations
tx = await wallet.build_deactivate_model(sender, model_id, gas=None)
tx = await wallet.build_set_model_commission_rate(sender, model_id, new_rate, gas=None)
tx = await wallet.build_report_model(sender, model_id, gas=None)
tx = await wallet.build_undo_report_model(sender, model_id, gas=None)
```

**Mining Submissions**

```python
# Submit data to fill a target
tx = await wallet.build_submit_data(
    sender,
    target_id,
    data_commitment,           # 32-byte hex
    data_url,                  # URL string
    data_checksum,             # 32-byte hex
    data_size,                 # int (bytes)
    model_id,                  # hex object ID
    embedding,                 # list[float]
    distance_score,            # float
    bond_coin,                 # {"id", "version", "digest"} dict
    gas=None,
)

# Claim rewards from a filled/expired target
tx = await wallet.build_claim_rewards(sender, target_id, gas=None)

# Report/undo-report a fraudulent submission
tx = await wallet.build_report_submission(sender, target_id, challenger=None, gas=None)
tx = await wallet.build_undo_report_submission(sender, target_id, gas=None)
```

**Challenges**

```python
# Initiate a challenge against a filled target
tx = await wallet.build_initiate_challenge(sender, target_id, bond_coin, gas=None)

# Validator reports that challenger is wrong
tx = await wallet.build_report_challenge(sender, challenge_id, gas=None)
tx = await wallet.build_undo_report_challenge(sender, challenge_id, gas=None)

# Resolve and claim challenge bond
tx = await wallet.build_claim_challenge_bond(sender, challenge_id, gas=None)
```

**Validator Management**

```python
tx = await wallet.build_add_validator(sender, pubkey_bytes, network_pubkey_bytes, worker_pubkey_bytes, net_address, p2p_address, primary_address, proxy_address, gas=None)
tx = await wallet.build_remove_validator(sender, pubkey_bytes, gas=None)
tx = await wallet.build_update_validator_metadata(sender, gas=None, next_epoch_network_address=None, ...)
tx = await wallet.build_set_commission_rate(sender, new_rate, gas=None)
tx = await wallet.build_report_validator(sender, reportee, gas=None)
tx = await wallet.build_undo_report_validator(sender, reportee, gas=None)
```

## End-to-End Example

```python
import asyncio
import json
from soma_sdk import SomaClient, WalletContext

async def transfer_soma():
    client = await SomaClient("http://localhost:9000")
    wallet = WalletContext("~/.soma/client.yaml")

    sender = await wallet.active_address()

    # Find a gas coin
    gas_objects = await wallet.get_gas_objects(sender)
    coin = json.loads(gas_objects[0])

    # Build, sign, and execute
    tx_bytes = await wallet.build_transfer_coin(
        sender=sender,
        recipient="0xRECIPIENT",
        coin=coin,
        amount=1_000_000_000,  # 1 SOMA
    )
    effects_json = await wallet.sign_and_execute_transaction(tx_bytes)
    print(json.loads(effects_json))

asyncio.run(transfer_soma())
```

## Building from Source

Requires Rust and Python ≥ 3.10.

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
