# Soma Refactor: Agent Marketplace

## Vision

Soma becomes a settlement and reputation ledger for a decentralized agent marketplace. The chain handles payments, and public reputation. Everything else (task orchestration, response delivery, agent self-improvement) lives off-chain.

The core thesis: **reputation adjusts pricing**. No Elo, no arbitration, no slashing. The platform exposes transaction history and seller ratings. Buyers with no history receive higher bids because it puts the seller's reputation at risk. Sellers without reputation are ranked lower by buyers. For a purchase the default rating is positive (implicit), unless the buyer updates the rating to be explicitly negative. The market does the rest.

---

## Token Economics

### Two-token model

**SOMA** — 1B total supply, shannons denomination (1B shannons per SOMA). Staking token. Validators stake SOMA to participate in consensus and earn emissions via geometric step-decay (high early, tapering off).

**USDC** is the payment and fee token. All payments and transaction fees (base_fee + value_fee) are denominated in USDC. Sellers earn USDC. Buyers pay USDC. All fees flow to the Protocol Fund.

This separation is the structural foundation. Protocol revenue is in a stable denomination — not inflated by token price or emissions. SOMA's value accrues from real economic demand, not circular fee-denominated-in-own-token reflexivity.

### CoinType

A single `CoinType` enum tags existing Coin objects. No new object type — same storage, same methods (`as_coin()`, `update_coin_balance()`), just a discriminator.

```rust
pub enum CoinType {
    SOMA,
    ETH_USDC,
}
```

`ObjectType::Coin` becomes `ObjectType::Coin(CoinType)`. The unified `Transfer` and `MergeCoins` transactions validate that all coins share the same `CoinType`. No mixing.

### USDC on-ramp


**Bridge (permissionless):** Anyone can deposit USDC on Ethereum and receive CoinType::Usdc on Soma. Withdrawals burn on Soma and release USDC on Ethereum. See the **USDC Bridge** section below for full design. This is the primary path for sellers to cash out.

### Protocol Fund

The Protocol Fund is a system-owned balance that accumulates ALL USDC fees — both `base_fee` (on every transaction) and `value_fee_bps` (on Payment). It has no automatic distribution logic in this version.

Future protocol upgrades add:
1. **On-chain DEX** — AMM or order book for SOMA/USDC trading
2. **Buyback-and-burn** — Protocol Fund executes continuous SOMA purchases via the DEX, purchased SOMA is burned. This creates non-inflationary demand for SOMA backed by real marketplace revenue.
3. **Fee tiers** — volume-based fee discounts, SOMA staking for additional discounts.

The sequencing matters: prove marketplace PMF first, then layer on tokenomics. Each upgrade is a protocol version bump.


### Revenue flows

```
All fees (USDC)
  │
  ├─ base_fee on every tx ──→ Protocol Fund (USDC)
  ├─ value_fee_bps on Payment ──→ Protocol Fund (USDC)
  │                                    └─ future: buyback-and-burn SOMA
  │
  └─ Payment remainder ──→ Seller (immediate)

Emissions (SOMA, inverse sqrt of total staked)
  │
  └─ C × √(total_staked) / epochs_per_year ──→ Validators (by stake weight)
      ~5% APY at 200M staked, self-adjusting
```

---

## On-Chain Primitives


### Payment



```rust
pub struct Receipt {
    pub id: ObjectId,             // ObjectID derived from tx_digest
    pub buyer: SomaAddress,
    pub seller: SomaAddress,
    pub coin: CoinType,
    pub amount: u64,
    pub rating: Rating,  // buyer's rating of seller (default: positive)
    pub rating_timeout: Epoch,
}

pub enum Rating {
    Positive,   // default — no tx needed. Pending + past deadline = Positive.
    Negative,   // buyer explicitly submitted Rate(negative)
}
```

---

## Transaction Types

```rust
// System
Genesis(GenesisTransaction),
ConsensusCommitPrologueV1(ConsensusCommitPrologueV1),
ChangeEpoch(ChangeEpoch),

// Validator management
AddValidator(AddValidatorArgs),
RemoveValidator(RemoveValidatorArgs),
ReportValidator { reportee: SomaAddress },
UndoReportValidator { reportee: SomaAddress },
UpdateValidatorMetadata(UpdateValidatorMetadataArgs),
SetCommissionRate { new_rate: u64 },

// Coins
TransferCoins {
    coins: Vec<ObjectRef>,              // one or more coins of the same CoinType
    amount: u64,          // recipient amount
    recipient: SomaAddress,       // one recipient
},

MergeCoins {
    coins: Vec<ObjectRef>,              // two or more coins of the same CoinType; first is the target
},

SplitCoins {
    coin: ObjectRef,
    amounts: Vec<u64>
}

TransferObjects { objects: Vec<ObjectRef>, recipient: SomaAddress },

// Validator staking (model staking removed)
AddStake { address: SomaAddress, coin_ref: ObjectRef, amount: Option<u64> },
WithdrawStake { staked_soma: ObjectRef },

Payment { // creates a receipt
    coins: Vec<ObjectRef>, // one or more coins of the same CoinType
    amount: u64, // per-recipient amount
    recipient: SomaAddress, // one recipient
}

Rate { receipt_id: ReceiptId, positive: Rating },

/// User initiates USDC withdrawal from Soma to Ethereum.
BridgeWithdraw(BridgeWithdrawArgs),

/// Certified USDC deposit from Ethereum. Gasless system tx submitted by bridge nodes.
BridgeDeposit(BridgeDepositArgs),

/// Emergency pause. Gasless system tx.
BridgeEmergencyPause(BridgeEmergencyPauseArgs),

/// Emergency unpause. Gasless system tx.
BridgeEmergencyUnpause(BridgeEmergencyUnpauseArgs),
```



## System State

SystemStateV1 currently contains:
- `marketplace_params: MarketplaceParameters`
- `protocol_fund_balance: u64` (accumulated USDC from ALL fees)
- `epoch`, `protocol_version`, `validators`, `parameters`
- `epoch_start_timestamp_ms`, `validator_report_records`
- `safe_mode`, `safe_mode_accumulated_fees`, `safe_mode_accumulated_emissions`
- `emission_pool: EmissionPool` (geometric step-decay)
- `bridge_state: BridgeState`

### EmissionPool — inverse square root of total staked

Emission rate is inversely proportional to the square root of total SOMA staked, following the same model as Ethereum and Hyperliquid. A single constant `C` (set in protocol config) defines the entire APY curve:

```
APY = C / √(total_staked)
emission_per_epoch = C × √(total_staked) / epochs_per_year
actual_emission = min(emission_per_epoch, pool.balance)
```

```rust
pub struct EmissionPool {
    /// Remaining balance from genesis allocation
    pub balance: u64,
}
```

The constant `C` is the only parameter, stored in protocol config and updatable via protocol version upgrade.

**Why inverse square root:**
- **Self-balancing.** More staking → lower APY per staker (discourages over-concentration). Less staking → higher APY (attracts validators). The market finds equilibrium without governance intervention.
- **One parameter.** Pick a target APY at an expected staking level, solve `C = target_apy × √(target_stake)`. The curve through all other staking levels is determined automatically.
- **Proven.** Ethereum (`BASE_REWARD_FACTOR = 64`) and Hyperliquid (`C ≈ 474`) both use this model. Real APY across major PoS chains clusters around 2-5% regardless of nominal rates — high-inflation chains (Cosmos 15-20%, Polkadot 12-15%) just erode returns with inflation. The sqrt model keeps emissions disciplined.
- **Sublinear total emission.** Total annual emission grows as `√(total_staked)`, not linearly. Doubling the stake increases total emission by only ~41%, so the pool drains slower than a flat-rate model under high participation.

**Why not flat amount or flat APY:**
- Flat amount per epoch: APY collapses as staking grows (inverse linear, too punishing). At high participation, validators leave. Requires manual protocol config updates to stay competitive.
- Flat APY: Total emissions scale linearly with staking. If 800M SOMA gets staked at 5%, that's 40M SOMA/year — pool drains unpredictably fast.

**Chosen constant: `C ≈ 707`** (targeting ~5% APY at 200M SOMA staked)

| Total Staked | APY | Annual Emission | Pool Life (~400M pool) |
|---|---|---|---|
| 50M | 10.0% | ~5M SOMA | ~80 years |
| 100M | 7.1% | ~7.1M SOMA | ~56 years |
| 200M | 5.0% | ~10M SOMA | ~40 years |
| 400M | 3.5% | ~14.1M SOMA | ~28 years |
| 600M | 2.9% | ~17.3M SOMA | ~23 years |

**Rationale for 5% at 200M staked:**
- Competitive with Ethereum (~3-5% real) and above Hyperliquid (~2.4%) to compensate for Soma being an earlier-stage chain without supplementary staking perks (Hyperliquid offers trading fee discounts to stakers).
- Soma validators earn SOMA emissions only — no fee income. The buyback-and-burn mechanism (Protocol Fund buys SOMA with USDC fees, burns it) is what creates deflationary pressure, but this only kicks in once marketplace revenue is meaningful. Until then, staking APY is the sole validator incentive and must be attractive on its own.
- At 200M staked (20% of supply), the pool lasts ~40 years — ample runway to establish marketplace revenue before emissions exhaust.
- `C` can be lowered via protocol upgrade once buyback-and-burn is live and SOMA has organic demand beyond staking yield.

Emissions go entirely to validators by stake weight.

---

## Fee Structure

Two fee types, two tokens:

| Fee | Token | Description |
|---|---|---|
| **base_fee** | USDC | Gas fee on every transaction. Paid by tx sender. Collected into Protocol Fund. |
| **value_fee_bps** | USDC | Basis points on payment amount, deducted on Payment. Collected into Protocol Fund. |

| Transaction | Fees |
|---|---|
| Payment | base_fee + value_fee_bps on amount |
| Rate | base_fee |
| Transfer | base_fee |
| MergeCoins | base_fee |
| BridgeDeposit | none (gasless system tx) |
| BridgeWithdraw | base_fee |
| BridgeEmergencyPause | none (gasless system tx) |
| BridgeEmergencyUnpause | none (gasless system tx) |

On Payment, the value fee is deducted from the buyer's USDC payment before settlement:
```
buyer pays: amount (USDC)
  ├─ value_fee = amount * value_fee_bps / 10_000 ──→ Protocol Fund
  └─ net       = amount - value_fee               ──→ Seller (immediate)
```

The seller receives `amount - value_fee` immediately on Payment — no escrow, no Deliver step. The fee is transparent to both parties at payment time.

## Reputation Graph

The reputation graph is fully derivable from receipt history. The chain stores raw data; the indexer computes derived metrics.

### Raw data (on-chain, stored in receipts)

For every receipt:
- buyer address, seller address (the edge)
- amount (edge weight — sybil defense, expensive to fake)
- rating (buyer's quality signal — default Positive, only Negative on-chain)
- coin type (SOMA or USDC)

Buyer reputation is implicit: the pattern of who they pay, how often, at what volume, and counterparty diversity.

### Derived metrics (computed by indexer, served by GraphQL)

Per seller:
- Total payments received
- Lifetime approval rate (% of receipts without negative rating)
- Total volume earned

Per buyer:
- Total payments made
- Total volume spent
- Counterparty diversity (unique sellers)

---

## Remaining Work

### Payment and Rate transactions

Implement `Payment` and `Rate` transaction types:
- Create `types/src/receipt.rs` with `Receipt`, `ReceiptId`, `Rating` types
- Add `Payment` and `Rate` variants to `TransactionKind`
- Add `Receipt` variant to `ObjectType`
- Create `authority/src/execution/payment.rs` with `PaymentExecutor`
- Add `payment()` and `rate()` methods to SDK
- Add `soma pay` and `soma rate` CLI commands
- Add receipt indexer pipeline and RPC endpoints
- Add e2e tests

### Ethereum bridge contract

The bridge-node crate exists with 34 tests, and the on-chain bridge executor is implemented with real ECDSA verification. Remaining:

- **Ethereum contract** (Foundry project in `bridge/evm/`). 3 Solidity files adapting Sui's patterns. Message encoding in Solidity MUST match `types::bridge::encode_bridge_message()`. The `test_message_encoding_known_values` test in `bridge-node/src/types.rs` provides exact byte-level verification.
- **Full deposit round-trip test** — mock Ethereum deposit event → EthSyncer emits → bridge node signs → gRPC exchange → `BridgeDeposit` system tx submitted.
- **Full withdrawal round-trip test** — `BridgeWithdraw` tx → checkpoint watcher detects `PendingWithdrawal` → signs → mock Ethereum contract call.
- **Committee rotation test** — epoch boundary detected → new validator set → sign committee update → mock Ethereum `updateCommitteeWithSignatures` call.

### Deployment

- Deploy Ethereum bridge contract (Sepolia testnet)
- Fresh testnet genesis with bridge committee, SOMA allocation to validators
- Deploy bridge nodes alongside each validator
- Update K8s manifests (validators + bridge nodes + indexer + graphql)
- Run DB migrations on indexer Postgres
- Smoke test bridge: deposit USDC on Ethereum, verify minted on Soma, withdraw back

---

## USDC Bridge

Bidirectional USDC bridge between Ethereum and Soma. Operated by validators as a protocol responsibility. Modeled after Sui's native bridge architecture: separate ECDSA keys for EVM compatibility, off-chain signature collection, single certified action submitted on-chain.

### Design principles

- **Validator-operated**: the bridge committee IS the validator set. Each validator runs a bridge node alongside their consensus node. No third-party relayers.
- **ECDSA keys for EVM**: each validator holds a Secp256k1 keypair for bridge signing, separate from their BLS consensus key. Solidity's `ecrecover` verifies these cheaply.
- **Off-chain aggregation**: bridge signatures are collected off-chain via gRPC between bridge nodes. Only the final certified action (with aggregated signatures) is submitted on-chain. No per-validator attestation transactions.
- **Tiered thresholds**: deposits and withdrawals require f+1 (~33%) stake. Emergency pause requires very low threshold (~5%). Emergency unpause requires 2/3 stake.

### Components

#### 1. Ethereum contract (Solidity)

Holds all real USDC. Deployed as an upgradable proxy.

```solidity
// Core operations
function deposit(bytes32 somaRecipient, uint256 amount) external;
function withdraw(
    bytes[] calldata signatures,
    address recipient,
    uint256 amount,
    uint64 nonce
) external;

// Emergency
function emergencyPause(bytes[] calldata signatures) external;  // low threshold
function emergencyUnpause(bytes[] calldata signatures) external; // high threshold

// Committee management (for validator set changes at epoch boundary)
function updateCommittee(
    bytes[] calldata signatures,
    address[] calldata newMembers,
    uint64[] calldata newVotingPowers
) external;
```

#### 2. Soma on-chain (implemented)

Bridge state, transaction types, and execution are implemented in:
- `types/src/bridge.rs` — BridgeState, BridgeCommittee, BridgeMember, PendingWithdrawal, message encoding, signing utilities
- `authority/src/execution/bridge.rs` — BridgeExecutor with real ECDSA ecrecover verification

#### 3. Bridge node (partially implemented)

`bridge-node/` crate with 9 modules and 34 tests:
- `eth_client.rs` — Raw JSON-RPC via reqwest with multi-endpoint rotation
- `eth_syncer.rs` — Finalized block poller + event listener with range-halving retry
- `server.rs` — gRPC signature collection with quorum detection
- `checkpoint_watcher.rs` — Scans checkpoints for PendingWithdrawal objects
- `node.rs` — Orchestrator (local signing works; peer gRPC exchange and quorum-based submission are TODO)
- `retry.rs` — Exponential backoff for transient errors

### Validator key management

Each validator holds three keypairs:
- **BLS12-381**: consensus protocol key (existing)
- **Ed25519**: network/worker key (existing)
- **Secp256k1**: bridge signing key — registered in genesis and updateable via validator metadata

---


## Protocol Version and Deployment

This is a breaking protocol change. Requires a new protocol version and a **fresh testnet**.

### Testnet (this refactor)

- Bump protocol version in `protocol-config`
- New genesis config with bridge committee (validator ECDSA keys), SOMA allocation to validators
- No USDC at genesis — all USDC enters via the bridge
- USDC bridge live from day one — deploy Ethereum contract, validators run bridge nodes
- `value_fee_bps` active from day one — fees accumulate in Protocol Fund
- Sellers earn real USDC via Payment transactions, can withdraw off-chain via bridge

### Mainnet (future protocol upgrades, in order)

1. **On-chain DEX** — AMM or order book for SOMA/USDC pair. Enables price discovery and the buyback mechanism.
2. **Buyback-and-burn** — Protocol Fund continuously purchases SOMA via the DEX and burns it.
3. **Retroactive airdrop** — SOMA distributed to testnet participants weighted by receipt graph metrics (volume, counterparty diversity, approval quality, longevity).
4. **Fee tiers** — volume-based discounts, SOMA staking for additional discounts.
5. **Bridge governance** — blocklisting, transfer limits, rate limiting, dispute windows.

---