# Soma Transaction Reference

Complete reference for all transaction types, object types, and execution logic.

---

## Object Types

| ObjectType | Owner | Description |
|---|---|---|
| `SystemState` | Shared | Global blockchain state (one per chain). Validators, epoch, emission pool, bridge state, protocol fund, parameters. |
| `Coin(CoinType::Soma)` | AddressOwner | SOMA token balance. Used for gas, staking, transfers. |
| `Coin(CoinType::Usdc)` | AddressOwner | USDC token balance. Minted via bridge deposits, used for payments. |
| `StakedSoma` | AddressOwner | Staked SOMA tokens delegated to a validator. Created on `AddStake`, destroyed on `WithdrawStake`. |
| `PendingWithdrawal` | Immutable | USDC withdrawal request waiting for bridge nodes to process. Created on `BridgeWithdraw`, observed by bridge nodes in checkpoints. |
| `Receipt` | Shared | **(planned)** Payment receipt between buyer and seller. Created on `Payment`, mutated on `Rate`. |

---

## Transaction Types

### System Transactions (gasless)

These are created by the protocol itself, not by users. They skip gas preparation entirely.

#### Genesis

Initializes the blockchain state at network launch.

| Field | Type | Description |
|---|---|---|
| `objects` | `Vec<Object>` | All initial objects (SystemState, validator stakes, initial coins) |

**Execution:** Creates all provided objects in the store. Run exactly once.

---

#### ConsensusCommitPrologueV1

Records consensus round metadata in the blockchain. Created automatically by the consensus layer.

**Execution:** No-op — the transaction exists for ordering and timestamp purposes.

---

#### ChangeEpoch

Transitions the network to a new epoch. Run by each validator at epoch boundary.

| Field | Type | Description |
|---|---|---|
| `epoch` | `EpochId` | Next epoch number |
| `epoch_start_timestamp_ms` | `u64` | Timestamp for new epoch start |
| `protocol_version` | `ProtocolVersion` | Protocol version for new epoch |
| `fees` | `u64` | Total fees collected in previous epoch |
| `epoch_randomness` | `Vec<u8>` | Randomness for new epoch |

**Execution:**
1. Loads SystemState
2. Clones state as backup
3. Calls `advance_epoch()`:
   - Distributes validator rewards (SOMA emissions by stake weight)
   - Processes validator set changes (joins, leaves, reports, slashing)
   - Updates emission pool (geometric step-decay)
   - Rotates committee and voting power
4. Creates `StakedSoma` reward objects for each validator
5. **Safe mode fallback:** If `advance_epoch()` fails, restores backup and calls `advance_epoch_safe_mode()` — minimal state bump, no rewards, no targets. Network continues in degraded mode.

**Objects created:** `StakedSoma` per validator (rewards)
**Objects modified:** `SystemState`

---

### Validator Management (gasless)

All validator transactions modify only `SystemState`. They are sent by validators and skip gas.

#### AddValidator

Registers a new validator in the pending set.

| Field | Type | Description |
|---|---|---|
| `pubkey_bytes` | `Vec<u8>` | BLS12-381 protocol public key |
| `network_pubkey_bytes` | `Vec<u8>` | Ed25519 network key |
| `worker_pubkey_bytes` | `Vec<u8>` | Ed25519 worker key |
| `proof_of_possession` | `Vec<u8>` | BLS signature proving key ownership |
| `net_address` | `Vec<u8>` | Consensus network address |
| `p2p_address` | `Vec<u8>` | P2P communication address |
| `primary_address` | `Vec<u8>` | Primary client-facing address |
| `proxy_address` | `Vec<u8>` | Proxy/data serving address |

**Validation:** Keys and PoP verified. Duplicate check. Addresses must be well-formed.
**Effect:** Validator added to pending set; becomes active at next epoch boundary.

---

#### RemoveValidator

Removes a validator from the active set.

| Field | Type | Description |
|---|---|---|
| `pubkey_bytes` | `Vec<u8>` | Public key of validator to remove |

**Validation:** Sender must be the validator being removed.
**Effect:** Validator marked for removal; exits at next epoch boundary.

---

#### ReportValidator

Reports a validator for misbehavior.

| Field | Type | Description |
|---|---|---|
| `reportee` | `SomaAddress` | Address of misbehaving validator |

**Validation:** Sender must be a validator. Cannot self-report. Reportee must exist.
**Effect:** Report count incremented. If quorum reached, reportee is slashed at epoch boundary.

---

#### UndoReportValidator

Retracts a previous validator report.

| Field | Type | Description |
|---|---|---|
| `reportee` | `SomaAddress` | Address to retract report for |

**Validation:** Sender must have previously reported this validator.
**Effect:** Report count decremented.

---

#### SetCommissionRate

Updates a validator's commission rate.

| Field | Type | Description |
|---|---|---|
| `new_rate` | `u64` | New rate in basis points (0–10000) |

**Validation:** Sender must be active validator. Rate ≤ 10000.
**Effect:** Commission rate updated (takes effect next epoch).

---

#### UpdateValidatorMetadata

Updates a validator's network addresses and/or keys.

| Field | Type | Description |
|---|---|---|
| `next_epoch_network_address` | `Option<Vec<u8>>` | New network address |
| `next_epoch_p2p_address` | `Option<Vec<u8>>` | New P2P address |
| `next_epoch_primary_address` | `Option<Vec<u8>>` | New primary address |
| `next_epoch_proxy_address` | `Option<Vec<u8>>` | New proxy address |
| `next_epoch_protocol_pubkey` | `Option<Vec<u8>>` | New BLS protocol key |
| `next_epoch_worker_pubkey` | `Option<Vec<u8>>` | New Ed25519 worker key |
| `next_epoch_network_pubkey` | `Option<Vec<u8>>` | New Ed25519 network key |
| `next_epoch_proof_of_possession` | `Option<Vec<u8>>` | PoP for new protocol key |

**Validation:** Sender must be active validator. Addresses well-formed. PoP valid if new key provided.
**Effect:** Metadata staged for next epoch.

---

### Coin & Object Transactions (user, requires gas)

#### TransferCoins

Sends coins to a recipient.

| Field | Type | Description |
|---|---|---|
| `coins` | `Vec<ObjectRef>` | One or more coins of the same `CoinType` |
| `amount` | `u64` | Amount to send |
| `recipient` | `SomaAddress` | Recipient address |

**Validation:**
- All coins must be same `CoinType`
- Sender must own all coins
- Total coin balance ≥ amount

**Execution:**
1. Merge all input coins into the primary coin (first coin, or gas coin if present)
2. Delete non-primary coins
3. Create new coin for recipient with specified amount
4. Primary coin retains remaining balance

**Fees:**
- `value_fee`: `bps_mul(amount, value_fee_bps)`
- `write_fee`: 2 (primary coin + recipient coin)

**Objects created:** One `Coin` for recipient
**Objects modified:** Primary coin (remaining balance)
**Objects deleted:** All non-primary input coins

---

#### MergeCoins

Combines multiple coins into one.

| Field | Type | Description |
|---|---|---|
| `coins` | `Vec<ObjectRef>` | Two or more coins of the same `CoinType`; first is the target |

**Validation:** All coins same `CoinType`. Sender owns all. No overflow.
**Execution:** Sum all balances into first coin. Delete the rest.
**Fees:** Base fee + write fees. No value fee.

**Objects modified:** First coin (updated balance)
**Objects deleted:** All coins except first

---

#### SplitCoins

Splits a coin into multiple new coins with specified amounts.

| Field | Type | Description |
|---|---|---|
| `coin` | `ObjectRef` | Source coin to split |
| `amounts` | `Vec<u64>` | Amounts for each new coin |

**Validation:**
- Sender must own coin
- Sum of amounts ≤ coin balance
- No arithmetic overflow

**Execution:**
1. Deduct sum of all amounts from source coin
2. Create one new coin per amount, owned by sender

**Fees:** Base fee + write fees. No value fee.

**Objects created:** One `Coin` per amount entry
**Objects modified:** Source coin (balance reduced)

---

#### TransferObjects

Transfers arbitrary (non-coin) objects to a new owner.

| Field | Type | Description |
|---|---|---|
| `objects` | `Vec<ObjectRef>` | Objects to transfer |
| `recipient` | `SomaAddress` | New owner |

**Validation:** Sender owns all objects.
**Execution:** Changes owner field on each object.
**Fees:** Base fee + write fees. No value fee.

**Objects modified:** All input objects (owner changed)

---

### Staking Transactions (user, requires gas)

#### AddStake

Stakes SOMA with a validator.

| Field | Type | Description |
|---|---|---|
| `address` | `SomaAddress` | Validator to stake with |
| `coin_ref` | `ObjectRef` | SOMA coin to stake from |
| `amount` | `Option<u64>` | Amount to stake, or `None` for entire balance |

**Validation:**
- Coin must be `CoinType::Soma` (USDC rejected)
- Validator must be active
- Sufficient balance

**Execution:**
1. Deduct stake amount from coin (or delete if staking full balance)
2. Create `StakedSoma` object owned by sender
3. Update validator's staking pool in `SystemState`

**Fees:**
- `value_fee`: half rate — `bps_mul(amount, value_fee_bps / 2)`
- `write_fee`: 3 objects

**Objects created:** `StakedSoma`
**Objects modified:** Source coin, `SystemState`
**Objects deleted:** Source coin (if full balance staked and not gas coin)

---

#### WithdrawStake

Withdraws staked SOMA plus accrued rewards.

| Field | Type | Description |
|---|---|---|
| `staked_soma` | `ObjectRef` | `StakedSoma` object to withdraw |

**Validation:** Sender owns the `StakedSoma`.
**Execution:**
1. Delete `StakedSoma` object
2. Compute rewards based on staking duration and pool earnings
3. Create new SOMA coin with `principal + rewards`
4. Update validator's staking pool in `SystemState`

**Fees:**
- `value_fee`: half rate on principal amount

**Objects created:** `Coin(Soma)` with principal + rewards
**Objects modified:** `SystemState`
**Objects deleted:** `StakedSoma`

---

### Bridge Transactions

#### BridgeDeposit (gasless system tx)

Mints USDC on Soma after a verified Ethereum deposit. Submitted by bridge nodes once signature quorum is reached.

| Field | Type | Description |
|---|---|---|
| `nonce` | `u64` | Unique deposit nonce (prevents replay) |
| `eth_tx_hash` | `[u8; 32]` | Ethereum transaction hash |
| `recipient` | `SomaAddress` | Soma address to receive USDC |
| `amount` | `u64` | USDC amount to mint (microdollars) |
| `aggregated_signature` | `Vec<u8>` | Concatenated 65-byte ECDSA signatures |
| `signer_bitmap` | `Vec<u8>` | Bitmap of which committee members signed |

**Validation:**
- Bridge not paused
- Nonce not already processed
- ECDSA signature verification: for each set bit in bitmap, extract 65-byte signature, ecrecover public key via Keccak256, verify it matches registered committee member's `ecdsa_pubkey`
- Total signing stake ≥ `threshold_deposit` (~33%)

**Execution:**
1. Verify signatures against bridge committee
2. Mint new `Coin(Usdc)` for recipient
3. Record nonce in `processed_deposit_nonces`
4. Increment `total_bridged_usdc`

**Objects created:** `Coin(Usdc)`
**Objects modified:** `SystemState`

---

#### BridgeWithdraw (user tx, requires gas)

Burns USDC on Soma and creates a withdrawal request for bridge nodes to process on Ethereum.

| Field | Type | Description |
|---|---|---|
| `payment_coin` | `ObjectRef` | USDC coin to burn |
| `amount` | `u64` | Amount to withdraw |
| `recipient_eth_address` | `[u8; 20]` | Ethereum recipient address |

**Validation:**
- Bridge not paused
- Coin must be `CoinType::Usdc`
- Sender owns coin
- Balance ≥ amount

**Execution:**
1. Deduct amount from USDC coin (or delete if exact balance)
2. Create `PendingWithdrawal` (immutable object)
3. Increment `next_withdrawal_nonce`
4. Decrement `total_bridged_usdc` (saturating)

Bridge nodes observe the `PendingWithdrawal` in checkpoints, collect signatures, and submit `withdraw()` to the Ethereum contract.

**Objects created:** `PendingWithdrawal`
**Objects modified:** USDC coin, `SystemState`
**Objects deleted:** USDC coin (if balance reaches 0)

---

#### BridgeEmergencyPause (gasless system tx)

Pauses all bridge operations. Low threshold for safety.

| Field | Type | Description |
|---|---|---|
| `aggregated_signature` | `Vec<u8>` | Concatenated ECDSA signatures |
| `signer_bitmap` | `Vec<u8>` | Bitmap of signers |

**Validation:** Signing stake ≥ `threshold_pause` (~5%). Same ECDSA verification as BridgeDeposit.
**Execution:** Sets `bridge_state.paused = true`.

**Objects modified:** `SystemState`

---

#### BridgeEmergencyUnpause (gasless system tx)

Resumes bridge operations. High threshold for security.

| Field | Type | Description |
|---|---|---|
| `aggregated_signature` | `Vec<u8>` | Concatenated ECDSA signatures |
| `signer_bitmap` | `Vec<u8>` | Bitmap of signers |

**Validation:** Signing stake ≥ `threshold_unpause` (~67%). Same ECDSA verification.
**Execution:** Sets `bridge_state.paused = false`.

**Objects modified:** `SystemState`

---

### Planned Transactions (not yet implemented)

#### Payment

Creates a payment receipt between buyer and seller. Deducts value fee for Protocol Fund.

| Field | Type | Description |
|---|---|---|
| `coins` | `Vec<ObjectRef>` | One or more coins of the same `CoinType` |
| `amount` | `u64` | Payment amount |
| `recipient` | `SomaAddress` | Seller address |

**Planned execution:**
1. Merge input coins
2. Compute `value_fee = amount * value_fee_bps / 10_000`
3. Deduct `amount` from buyer's coin
4. Credit `amount - value_fee` to seller (new coin)
5. Credit `value_fee` to `protocol_fund_balance` in SystemState
6. Create `Receipt` shared object with default `Rating::Positive`

**Objects created:** `Receipt` (shared), `Coin` for seller
**Objects modified:** Buyer's coin, `SystemState`

---

#### Rate

Updates the rating on a payment receipt.

| Field | Type | Description |
|---|---|---|
| `receipt_id` | `ReceiptId` | Receipt to rate |
| `positive` | `Rating` | `Positive` or `Negative` |

**Planned execution:**
1. Load `Receipt` shared object
2. Validate sender is the buyer on the receipt
3. Validate within `rating_timeout` epoch window
4. Update `receipt.rating`

**Objects modified:** `Receipt`

---

## Execution Pipeline

Every transaction flows through this pipeline (defined in `authority/src/execution/mod.rs`):

```
1. EXECUTOR DISPATCH
   TransactionKind → specialized executor
   (CoinExecutor, ValidatorExecutor, BridgeExecutor, etc.)

2. GAS PREPARATION (user txs only)
   - Smash gas coins into one
   - Deduct base_fee (DOS protection)
   - System txs skip this entirely

3. INPUT LOADING
   - Owned objects loaded by ObjectRef
   - Shared objects loaded with assigned versions

4. EXECUTE
   executor.execute(&mut store, ...)
   On error: revert all non-gas changes

5. REMAINING FEES (user txs only)
   - Compute value_fee (BPS of amount) + write_fee (per object)
   - Deduct from gas coin
   - On failure: revert but keep base_fee deduction

6. OWNERSHIP INVARIANTS
   - Verify all mutable inputs still owned by sender
   - Ensure mutable shared objects have version bumped

7. GENERATE EFFECTS
   - Created/mutated/deleted objects
   - Gas cost breakdown
   - Execution status (success/failure)
```

## Fee Structure

| Fee | When | Calculation |
|---|---|---|
| `base_fee` | Every user tx | Flat fee, deducted during gas preparation |
| `value_fee` | TransferCoins, Payment, AddStake, WithdrawStake | `amount * value_fee_bps / 10_000` (half rate for staking) |
| `write_fee` | Every user tx | `write_object_fee * num_objects_written` |

System transactions (Genesis, ConsensusCommit, ChangeEpoch, validator management, bridge deposit/pause/unpause) are gasless.

## Constants

| Constant | Value | Description |
|---|---|---|
| `BPS_DENOMINATOR` | 10000 | 100% in basis points |
| `SHANNONS_PER_SOMA` | 1,000,000,000 | Smallest SOMA unit |
| `TOTAL_SUPPLY_SOMA` | 1,000,000,000 | 1 billion SOMA total supply |
