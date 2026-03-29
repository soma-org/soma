# Soma Refactor: Agent Marketplace

## Vision

Soma becomes a settlement and reputation ledger for a decentralized agent marketplace. The chain handles payments, the ask/bid order book, and public reputation. Everything else (task orchestration, response delivery, agent self-improvement) lives off-chain.

The core thesis: **reputation is pricing**. No Elo, no arbitration, no slashing. The platform exposes transaction history and seller ratings. Accepting a bid IS the buyer's endorsement — only negative seller ratings go on-chain; the default is positive. The market does the rest.

---

## Token Economics

### Two-token model

**SOMA** — 1B total supply, shannons denomination (1B shannons per SOMA). Gas and staking token. Validators stake SOMA to participate in consensus and earn emissions via geometric step-decay (high early, tapering off).

**USDC** is the marketplace payment token. All asks, bids, escrow, settlements, and seller vaults are denominated in USDC. Sellers earn USDC. Buyers pay USDC (indirectly, via the Glass service). The protocol collects marketplace fees in USDC.

This separation is the structural foundation. Protocol revenue is in a stable denomination — not inflated by token price or emissions. SOMA's value accrues from real economic demand, not circular fee-denominated-in-own-token reflexivity.

### CoinType

A single `CoinType` enum tags existing Coin objects. No new object type — same storage, same methods (`as_coin()`, `update_coin_balance()`), just a discriminator.

```rust
pub enum CoinType {
    Soma,
    Usdc,
}
```

`ObjectType::Coin` becomes `ObjectType::Coin(CoinType)`. The unified `Transfer` and `MergeCoins` transactions validate that all coins share the same `CoinType`. No mixing.

### USDC on-ramp

Two on-ramps, both available at launch:

**Bridge (permissionless):** Anyone can deposit USDC on Ethereum and receive CoinType::Usdc on Soma. Withdrawals burn on Soma and release USDC on Ethereum. See the **USDC Bridge** section below for full design. This is the primary path for sellers to cash out.

**Glass (SaaS model):** Agent consumers pay Glass via credit card for credits (off-chain, standard SaaS billing). When a consumer uses Glass to delegate a task, Glass spends its *own* on-chain USDC to create asks and accept bids on behalf of the consumer. The consumer never holds or touches crypto — they interact with Glass as a SaaS product. Glass holds USDC received via the bridge (same as any other user).

On-chain, the flow is: Glass address → escrow → seller vault. The protocol doesn't know or care that a credit card payment triggered it. Sellers earn real USDC and can bridge it out to Ethereum independently.

### Protocol Fund

The Protocol Fund is a system-owned balance that accumulates USDC from marketplace fees (`value_fee_bps` on AcceptBid). It has no automatic distribution logic in this version.

Future protocol upgrades add:
1. **On-chain DEX** — AMM or order book for SOMA/USDC trading
2. **Buyback-and-burn** — Protocol Fund executes continuous SOMA purchases via the DEX, purchased SOMA is burned. This creates non-inflationary demand for SOMA backed by real marketplace revenue.
3. **Fee tiers** — volume-based fee discounts, SOMA staking for additional discounts.

The sequencing matters: prove marketplace PMF first, then layer on tokenomics. Each upgrade is a protocol version bump.

### Retroactive distribution

There is no points system. The testnet settlement graph *is* the airdrop data.

On mainnet launch, SOMA is distributed to testnet participants weighted by:
- Total USDC volume transacted (as buyer and seller)
- Unique counterparties (graph breadth — anti-sybil)
- Approval rate from established counterparties (rating provenance quality)
- Longevity and consistency of activity

Anti-gaming: wash trading between colluding accounts is detectable via graph topology — low counterparty diversity, symmetric volume, repetitive task hashes. The reputation system's distinction between "established" and "new" rater provenance naturally down-weights collusion rings.

### Revenue flows

```
Marketplace activity (USDC)
  │
  ├─ value_fee_bps on AcceptBid ──→ Protocol Fund (USDC)
  │                                    └─ future: buyback-and-burn SOMA
  │
  └─ remainder ──→ SellerVault (immediate on AcceptBid)

Gas fees (SOMA)
  │
  └─ base_fee on every tx ──→ Validators (via epoch distribution)

Emissions (SOMA, geometric step-decay)
  │
  └─ epoch emission ──→ Validators (by stake weight)
      starts at 100K SOMA/epoch, decreases 10% every 10 epochs
```

---

## On-Chain Primitives

### Ask

A buyer's request for work.

```rust
pub struct Ask {
    pub id: AskId,                    // ObjectID derived from tx_digest
    pub buyer: SomaAddress,
    pub task_digest: TaskDigest,      // blake2b of task content (opaque to chain)
    pub max_price_per_bid: u64,       // USDC microdollars — per-bid price cap
    pub num_bids_wanted: u32,         // how many bids the buyer intends to accept
    pub timeout_ms: u64,              // deadline for bids
    pub created_at_ms: u64,           // consensus timestamp at creation
    pub status: AskStatus,
    pub accepted_bid_count: u32,      // how many bids accepted so far
}

pub enum AskStatus {
    Open,       // accepting bids
    Filled,     // accepted_bid_count == num_bids_wanted
    Cancelled,  // buyer cancelled before accepting any bids
    Expired,    // timeout elapsed, no bids accepted
}
```

`TaskDigest` is a proper newtype over `Digest`, following the same pattern as `TransactionDigest`, `ObjectDigest`, etc. in `digests.rs`. Base58 serialization, `FromStr`, `Display`, the full set of standard derives.

### Bid

A seller's offer to fulfill an ask.

```rust
pub struct Bid {
    pub id: BidId,                    // ObjectID derived from tx_digest
    pub ask_id: AskId,
    pub seller: SomaAddress,
    pub price: u64,                   // USDC microdollars, must be <= ask.max_price_per_bid
    pub response_digest: ResponseDigest, // blake2b of off-chain response content
    pub created_at_ms: u64,
    pub status: BidStatus,
}

pub enum BidStatus {
    Pending,    // waiting for buyer decision
    Accepted,   // buyer accepted, payment settled to seller vault
    Rejected,   // buyer picked other bids
    Expired,    // ask timeout elapsed without acceptance
}
```

`ResponseDigest` is a newtype over `Digest`, same pattern as `TaskDigest`. The seller includes the hash of their response when bidding — the buyer can verify the response content off-chain before accepting.

### Settlement

Created when a buyer accepts a bid. AcceptBid = settlement = payment. This is the core edge in the reputation graph.

Accepting a bid is itself the positive signal for the buyer — no separate buyer rating needed. Users generally don't bother rating; only negative seller ratings go on-chain. The default is positive.

```rust
pub struct Settlement {
    pub id: SettlementId,             // ObjectID derived from tx_digest
    pub ask_id: AskId,
    pub bid_id: BidId,
    pub buyer: SomaAddress,
    pub seller: SomaAddress,
    pub amount: u64,                  // actual payment in USDC microdollars (= bid.price - value_fee)
    pub task_digest: TaskDigest,      // copied from ask — grouping key for competitions
    pub response_digest: ResponseDigest, // copied from bid — auditability
    pub settled_at_ms: u64,           // consensus timestamp
    pub seller_rating: SellerRating,  // buyer's rating of seller (default: positive)
    pub rating_deadline_ms: u64,      // deadline for buyer to submit negative rating
}

pub enum SellerRating {
    Positive,   // default — no tx needed. Pending + past deadline = Positive.
    Negative,   // buyer explicitly submitted RateSeller(bad)
}
```

No `RateBuyer`. The buyer's "rating" is the accept itself — the settlement graph (who accepted whose bids, how often, at what volume) IS the buyer reputation data. The indexer computes buyer reputation from acceptance patterns.

### SellerVault

Per-seller USDC balance accumulator. Created lazily on first delivery. Prevents coin fragmentation — sellers withdraw in bulk whenever they choose.

```rust
pub struct SellerVault {
    pub id: ObjectID,
    pub owner: SomaAddress,
    pub balance: u64,              // USDC microdollars
}
```

---

## Transaction Types

### Keep (existing)

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

// Coins and objects
Transfer {
    coins: Vec<ObjectRef>,              // one or more coins of the same CoinType
    amounts: Option<Vec<u64>>,          // per-recipient amounts (None = send all, merging inputs)
    recipients: Vec<SomaAddress>,       // one or more recipients
},
MergeCoins {
    coins: Vec<ObjectRef>,              // two or more coins of the same CoinType; first is the target
},
TransferObjects { objects: Vec<ObjectRef>, recipient: SomaAddress },

// Validator staking (model staking removed)
AddStake { address: SomaAddress, coin_ref: ObjectRef, amount: Option<u64> },
WithdrawStake { staked_soma: ObjectRef },
```

### Cut (remove entirely)

```rust
CreateModel(CreateModelArgs),
CommitModel(CommitModelArgs),
RevealModel(RevealModelArgs),
AddStakeToModel { .. },
SetModelCommissionRate { .. },
DeactivateModel { .. },
ReportModel { .. },
UndoReportModel { .. },
SubmitData(SubmitDataArgs),
ClaimRewards(ClaimRewardsArgs),
ReportSubmission { .. },
UndoReportSubmission { .. },
```

### Add (new marketplace transactions)

```rust
/// Buyer posts a task request. Pays gas fee.
CreateAsk(CreateAskArgs),

/// Buyer cancels an open ask before any bids are accepted.
CancelAsk { ask_id: AskId },

/// Seller offers to fulfill an ask. Includes response_digest (hash of off-chain response).
/// Price must be <= ask.max_price_per_bid. Pays gas fee.
CreateBid(CreateBidArgs),

/// Buyer accepts a bid. This IS the settlement — immediate payment.
/// Deducts bid.price from payment_coin, takes value_fee to Protocol Fund,
/// credits remainder to seller's SellerVault (created lazily if needed).
/// Creates Settlement object. Can be called up to ask.num_bids_wanted times.
AcceptBid(AcceptBidArgs),

/// Buyer submits a negative rating for a seller on a settlement.
/// Must be within rating_deadline_ms. Only negative ratings go on-chain —
/// the default is positive (no tx needed). Pays gas fee.
RateSeller { settlement_id: SettlementId },

/// Seller withdraws accumulated USDC earnings from their vault into a coin.
WithdrawFromVault { vault: ObjectRef, amount: Option<u64>, recipient_coin: Option<ObjectRef> },

/// User initiates USDC withdrawal from Soma to Ethereum.
BridgeWithdraw(BridgeWithdrawArgs),
```

Note: No `Deliver` transaction. The seller's work happens off-chain. The buyer verifies the response (matching the bid's `response_digest`) off-chain, then accepts the bid. Accept = pay = settle. This eliminates a round-trip and removes the escrow complexity (no funds locked between accept and deliver, no expiry refund logic).

Note: No `RateBuyer`. Accepting a bid IS the positive signal. The settlement graph gives the indexer everything it needs to compute buyer reputation — who they accepted, how often, volume, counterparty diversity.

### Add (bridge system transactions — gasless)

```rust
/// Certified USDC deposit from Ethereum. Gasless system tx submitted by bridge nodes.
BridgeDeposit(BridgeDepositArgs),

/// Emergency pause. Gasless system tx.
BridgeEmergencyPause(BridgeEmergencyPauseArgs),

/// Emergency unpause. Gasless system tx.
BridgeEmergencyUnpause(BridgeEmergencyUnpauseArgs),
```

### Arg Structs

```rust
pub struct CreateAskArgs {
    pub task_digest: TaskDigest,       // blake2b of task content
    pub max_price_per_bid: u64,
    pub num_bids_wanted: u32,
    pub timeout_ms: u64,
}

pub struct CreateBidArgs {
    pub ask_id: AskId,
    pub price: u64,
    pub response_digest: ResponseDigest,  // blake2b of off-chain response content
}

pub struct AcceptBidArgs {
    pub ask_id: AskId,
    pub bid_id: BidId,
    pub payment_coin: ObjectRef,     // must be CoinType::Usdc
}
```

### Digest Types (new in `digests.rs`)

```rust
// Newtype wrappers over Digest, following the same pattern as TransactionDigest et al.
// Base58 serialization, FromStr, Display, Clone, Copy, PartialEq, Eq, Hash, Ord, etc.

/// Blake2b hash of task content (opaque to chain — content lives off-chain).
pub struct TaskDigest(Digest);

/// Blake2b hash of seller's response content (opaque to chain — content lives off-chain).
pub struct ResponseDigest(Digest);

/// Alias types for marketplace object IDs.
pub type AskId = ObjectID;
pub type BidId = ObjectID;
pub type SettlementId = ObjectID;
```

---

## Execution Logic

### CreateAsk

1. Validate timeout is within [min_ask_timeout_ms, max_ask_timeout_ms]
2. Validate max_price_per_bid > 0, num_bids_wanted > 0
3. Create Ask object (status: Open, accepted_bid_count: 0)

### CreateBid

1. Validate ask exists and status == Open
2. Validate current_time < ask.created_at_ms + ask.timeout_ms
3. Validate price <= ask.max_price_per_bid and price > 0
4. Validate seller != ask.buyer
5. Create Bid object (status: Pending) with seller's response_digest

### AcceptBid

AcceptBid is the settlement event — accept, pay, and settle in one atomic transaction. No escrow, no Deliver round-trip.

1. Validate ask exists, sender == ask.buyer
2. Validate bid exists, bid.ask_id == ask_id, bid.status == Pending
3. Validate ask.accepted_bid_count < ask.num_bids_wanted
4. Validate payment_coin is CoinType::Usdc
5. Compute value_fee = bid.price * marketplace_params.value_fee_bps / 10_000
6. Deduct bid.price from payment_coin (full amount from buyer)
7. Credit value_fee to system_state.protocol_fund_balance
8. Load or create seller's SellerVault
9. Credit vault.balance += bid.price - value_fee (immediate payment to seller)
10. Set bid.status = Accepted
11. Increment ask.accepted_bid_count
12. Create Settlement object:
    - Copy buyer, seller, task_digest from ask; response_digest from bid
    - amount = bid.price - value_fee
    - seller_rating: Positive (default — buyer must explicitly submit RateSeller to change)
    - rating_deadline_ms = current_time + rating_window_ms
13. If accepted_bid_count == num_bids_wanted: set ask.status = Filled

### CancelAsk

1. Validate ask exists, sender == ask.buyer
2. Validate ask.accepted_bid_count == 0 (can't cancel after accepting bids)
3. Set ask.status = Cancelled

### RateSeller

Only negative ratings go on-chain. The default is Positive (no tx needed).

1. Validate settlement exists, sender == settlement.buyer
2. Validate settlement.seller_rating == Positive (not already rated Negative)
3. Validate current_time < settlement.rating_deadline_ms
4. Set settlement.seller_rating = Negative

### WithdrawFromVault

1. Validate vault exists, sender == vault.owner
2. If amount specified: validate amount <= vault.balance, deduct amount
3. If no amount: withdraw full balance
4. If recipient_coin provided: credit the coin. Otherwise create new coin owned by sender.
5. If vault.balance == 0 after withdrawal: optionally delete vault object to reclaim storage

### Expiry (lazy — checked when objects are accessed, or at epoch boundary)

- **Asks** past timeout with no accepted bids: status -> Expired
- **Pending bids** past ask timeout without acceptance: status -> Expired
- **Seller ratings** past deadline: remain Positive (the default — no state mutation needed)

No escrow refund logic. Since AcceptBid = immediate payment, there's no locked-up USDC to reclaim. Expiry is purely a status cleanup concern. No `ClaimExpiredEscrow` transaction needed.

---

## System State Changes

### Remove from SystemStateV1

```rust
pub model_registry: ModelRegistry,    // entire model system
pub target_state: TargetState,        // target generation and difficulty
```

### Add to SystemStateV1

```rust
pub marketplace_params: MarketplaceParameters,
pub protocol_fund_balance: u64,       // accumulated USDC from value fees
```

```rust
pub struct MarketplaceParameters {
    /// Rating window in milliseconds (e.g., 48 hours = 172_800_000)
    pub rating_window_ms: u64,
    /// Minimum ask timeout (floor, e.g., 10 seconds = 10_000)
    pub min_ask_timeout_ms: u64,
    /// Maximum ask timeout (ceiling, e.g., 7 days = 604_800_000)
    pub max_ask_timeout_ms: u64,
    /// Marketplace fee in basis points (e.g., 250 = 2.5%)
    pub value_fee_bps: u64,
}
```

### Keep unchanged

- `epoch`, `protocol_version`, `validators`, `parameters`
- `epoch_start_timestamp_ms`, `validator_report_records`
- `safe_mode`, `safe_mode_accumulated_fees`, `safe_mode_accumulated_emissions`

### EmissionPool — rewrite to geometric step-decay (Sui's StakeSubsidy model)

Replace the current flat `emission_per_epoch` with Sui's logarithmic decay:

```rust
pub struct EmissionPool {
    /// Remaining balance from genesis allocation
    pub balance: u64,
    /// Count of epochs emissions have been distributed
    pub distribution_counter: u64,
    /// Current emission amount per epoch (decays over time)
    pub current_distribution_amount: u64,
    /// Number of epochs per decay period
    pub period_length: u64,
    /// Decay rate in basis points (e.g., 1000 = 10% decrease per period)
    pub decrease_rate: u16,
}
```

Each epoch: emit `min(current_distribution_amount, balance)`. Every `period_length` epochs, reduce `current_distribution_amount` by `decrease_rate` bps. This creates geometric decay — high early emissions tapering off asymptotically.

**Suggested parameters (tunable in genesis config):**
- `current_distribution_amount`: 100,000 SOMA/epoch (0.01% of 1B supply)
- `period_length`: 10 epochs
- `decrease_rate`: 1000 bps (10% reduction per period)
- Epoch 0-9: 100K SOMA/epoch → Epoch 10-19: 90K → Epoch 20-29: 81K → ...

Emissions go entirely to validators by stake weight. Remove target reward allocation, model reward splits, claimer incentives.

---

## Fee Structure

Two fee types, two tokens:

| Fee | Token | Description |
|---|---|---|
| **base_fee** | SOMA | Gas fee on every transaction. Paid by tx sender. Distributed to validators at epoch boundary. |
| **value_fee_bps** | USDC | Basis points on `bid.price`, deducted on AcceptBid. Collected into Protocol Fund. Primary protocol revenue. |

| Transaction | Fees |
|---|---|
| CreateAsk | base_fee |
| CancelAsk | base_fee |
| CreateBid | base_fee |
| AcceptBid | base_fee + value_fee_bps on bid.price |
| RateSeller | base_fee |
| WithdrawFromVault | base_fee |
| Transfer | base_fee |
| MergeCoins | base_fee |
| BridgeDeposit | none (gasless system tx) |
| BridgeWithdraw | base_fee |
| BridgeEmergencyPause | none (gasless system tx) |
| BridgeEmergencyUnpause | none (gasless system tx) |

On AcceptBid, the value fee is deducted from the buyer's USDC payment before settlement:
```
buyer pays: bid.price (USDC)
  ├─ value_fee = bid.price * value_fee_bps / 10_000 ──→ Protocol Fund
  └─ net       = bid.price - value_fee                ──→ SellerVault (immediate)
```

The seller receives `bid.price - value_fee` immediately on AcceptBid — no escrow, no Deliver step. The fee is transparent to both parties at acceptance time.

---

## Epoch Transition

### Remove

- Difficulty adjustment (target_state EMA, threshold updates)
- Target generation (initial targets, spawn-on-fill)
- Model reveal processing (commit -> active transitions)
- Model staking reward distribution
- Target reward calculation and allocation

### Keep

- Validator reward distribution (receives all SOMA emissions + accumulated SOMA gas fees)
- Validator set rotation and report processing
- Safe mode handling

Note: USDC value fees accumulate in `protocol_fund_balance` in real time (on each AcceptBid), not at epoch boundary. Validators receive SOMA only.

### Add

- Nothing strictly required. Expiry is handled lazily or via explicit user transactions.
- Optional: epoch boundary could sweep obviously-expired asks to keep state clean, but not required for correctness.

---

## Reputation Graph

The reputation graph is fully derivable from settlement history. The chain stores raw data; the indexer computes derived metrics.

### Raw data (on-chain, stored in settlements)

For every settlement:
- buyer address, seller address (the edge)
- amount (edge weight — sybil defense, expensive to fake)
- task_digest (grouping key — settlements sharing a task_digest were competing)
- seller_rating (buyer's quality signal — default Positive, only Negative on-chain)
- timestamp (recency weighting)

Buyer reputation is implicit: the pattern of who they accept, how often, at what volume, and counterparty diversity. No explicit buyer rating needed — the settlement graph IS the buyer's reputation.

### Derived metrics (computed by indexer, served by GraphQL)

Per seller:
- Total deliveries (bids accepted)
- Lifetime approval rate (% of settlements without negative rating)
- Rolling 30-day approval rate
- Bid-to-win ratio (bids submitted vs bids accepted)
- Total volume earned

Per buyer:
- Total asks created
- Total bids accepted
- Total volume spent
- Counterparty diversity (unique sellers)
- Average acceptance rate per ask (bids accepted / bids received)

Per address pair (counterparty edges):
- Transaction count between the pair
- Total volume between the pair
- Seller approval rate from this buyer

Rating provenance (critical for sybil resistance):
- Breakdown: negative ratings from established buyers (>N transactions, >X volume) vs ratings from new accounts
- This lets clients weight ratings by rater quality — a negative from a high-volume buyer matters more than one from a fresh account

Competition grouping:
- Settlements sharing a task_digest indicate competitive evaluation
- Which sellers' bids were accepted vs rejected for the same task is a strong signal

---

## Crate-by-Crate Changes

### `types` — heavy surgery

| File | Action | Details |
|---|---|---|
| `transaction.rs` | **Rewrite** | Remove model/submission/target variants + arg structs. Replace `TransferCoin`/`PayCoins` with unified `Transfer` + `MergeCoins`. Add marketplace variants (CreateAsk, CancelAsk, CreateBid, AcceptBid, RateSeller, WithdrawFromVault) + bridge variants + arg structs using `TaskDigest`/`ResponseDigest`. Keep system/validator/staking variants. |
| `target.rs` | **Delete** | |
| `model.rs` | **Delete** | |
| `submission.rs` | **Delete** | |
| `tensor.rs` | **Delete** | SomaTensor only used for embeddings/distances |
| `metadata.rs` | **Delete** | Model weight manifests, no longer needed |
| `crypto.rs` | **Simplify** | Remove DecryptionKey, DecryptionKeyCommitment, EmbeddingCommitment, ModelWeightsCommitment. Keep Ed25519/BLS signing. |
| `digests.rs` | **Simplify** | Remove model/embedding/key commitment types. Add `TaskDigest`, `ResponseDigest` as newtype wrappers over `Digest` (same pattern as `TransactionDigest` — Base58, FromStr, Display, full derives). Add `AskId`, `BidId`, `SettlementId` as ObjectID type aliases. |
| `object.rs` | **Modify** | Add `CoinType` enum. Change `ObjectType::Coin` to `ObjectType::Coin(CoinType)`. Add object type variants for Ask, Bid, Settlement, SellerVault. |
| `system_state/mod.rs` | **Modify** | Remove `model_registry`, `target_state`. Add `marketplace_params`, `protocol_fund_balance`, `bridge_state`. Update `create()`, `advance_epoch()`. |
| `system_state/model_registry.rs` | **Delete** | |
| `system_state/target_state.rs` | **Delete** | |
| `system_state/emission.rs` | **Rewrite** | Replace flat `emission_per_epoch` with geometric step-decay (Sui's StakeSubsidy model). Add `distribution_counter`, `period_length`, `decrease_rate`. |
| `system_state/staking.rs` | **Simplify** | Remove model staking pool references. Keep validator staking. |
| `system_state/validator.rs` | **Modify** | Add Secp256k1 bridge key to ValidatorInfo |
| `system_state/epoch_start.rs` | **Keep** | |
| `config/genesis_config.rs` | **Modify** | Remove model/target genesis params. Change `TOTAL_SUPPLY_SOMA` from 10M to 1B. Add marketplace_params, bridge committee config (validator ECDSA keys), emission decay params. No USDC at genesis — all USDC enters via bridge. |
| **New: `ask.rs`** | **Create** | Ask, AskId, AskStatus |
| **New: `bid.rs`** | **Create** | Bid, BidId, BidStatus |
| **New: `settlement.rs`** | **Create** | Settlement, SettlementId, Rating (replaces old submission.rs) |
| **New: `vault.rs`** | **Create** | SellerVault |
| **New: `bridge.rs`** | **Create** | BridgeState, BridgeCommittee, BridgeMember, PendingWithdrawal, bridge tx arg structs |

### `protocol-config`

**Remove:**
- All target params: distance threshold, initial targets, hits per epoch, EMA decay, target reward allocation bps, submitter/model/claimer reward split bps
- All model params: architecture version, commit/reveal windows, model minimum stake
- All submission params: bond per byte, max data size
- `SomaTensor` type (used for target thresholds)

**Modify:**
- `TOTAL_SUPPLY_SOMA`: 10_000_000 → 1_000_000_000 (10M → 1B)

**Add:**
- `rating_window_ms: u64`
- `min_ask_timeout_ms: u64`
- `max_ask_timeout_ms: u64`
- `value_fee_bps: u64`
- `emission_initial_distribution_amount: u64`
- `emission_period_length: u64`
- `emission_decrease_rate: u16`

### `authority/src/execution/`

| File | Action | Details |
|---|---|---|
| `model.rs` | **Delete** | |
| `submission.rs` | **Delete** | |
| `staking.rs` | **Simplify** | Remove AddStakeToModel handling |
| `change_epoch.rs` | **Simplify** | Remove target reward calc, model reward distribution, difficulty adjustment. All emissions + fees → validators. |
| `mod.rs` | **Modify** | Update `create_executor()`: remove Model/Submission arms, add Marketplace and Bridge arms |
| `coin.rs` | **Rewrite** | Replace `TransferCoin`/`PayCoins` with unified `Transfer` (multi-coin input, multi-recipient output, CoinType validation — no mixing) and `MergeCoins` (combine N coins of same CoinType into first). Delete old executor paths. |
| **New: `marketplace.rs`** | **Create** | MarketplaceExecutor: CreateAsk, CancelAsk, CreateBid, AcceptBid (settlement + value_fee deduction + SellerVault credit), RateSeller (negative only), WithdrawFromVault |
| **New: `bridge.rs`** | **Create** | BridgeExecutor: BridgeDeposit (verify ECDSA sigs, mint USDC), BridgeWithdraw (burn USDC, create PendingWithdrawal), BridgeEmergencyPause/Unpause |

### `authority/src/authority.rs`

**Remove:** Target generation at epoch boundary, model reveal processing.

### Crates and directories to delete

| Crate/Directory | Reason |
|---|---|
| `scoring` | Model evaluation engine, no models |
| `models` | V1 transformer architecture, no models |
| `arrgen` | Array generation for model architectures (verify nothing else depends on it) |
| `python-sdk/` | PyO3 wrapper — CLI replaces all functionality. Removes PYO3_PYTHON/maturin build complexity. |
| `python-examples/` | SDK usage examples — no SDK to exemplify |
| `python-models/` | Python model code — no models |

### `sdk` — rewrite API surface

**Remove:**
- `score()`, `submit_data()`, `claim_rewards()`
- `create_model()`, `commit_model()`, `reveal_model()`
- `add_stake_to_model()`, `set_model_commission_rate()`, `deactivate_model()`
- `transfer_coin()`, `pay_coins()` (replaced by unified `transfer()`)

**Keep:**
- `add_stake()`, `withdraw_stake()` (validator only)
- Connection/signing infrastructure

**Add:**
- `transfer(coins, amounts?, recipients)` — unified transfer with multi-coin input, multi-recipient output
- `merge_coins(coins)` — merge N coins of same CoinType into first
- `create_ask(task_digest, max_price_per_bid, num_bids_wanted, timeout_ms)`
- `cancel_ask(ask_id)`
- `create_bid(ask_id, price, response_digest)`
- `accept_bid(ask_id, bid_id, payment_coin)`
- `rate_seller(settlement_id)` — negative rating only (no good/bad bool)
- `withdraw_from_vault(vault, amount?)`
- `get_open_asks(filters?)`
- `get_bids_for_ask(ask_id)`
- `get_settlements(address, role?)`
- `get_reputation(address)`
- `get_protocol_fund()` — current Protocol Fund USDC balance

### `python-sdk` — **delete entirely**

The Python SDK is removed from the workspace. Everything the SDK does is available via the CLI, which is the primary interface for both humans and agents. An agent calling Soma will shell out to the CLI (or use the RPC directly). The PyO3 wrapper adds build complexity (PYO3_PYTHON, maturin, Python version pinning) and maintenance burden for zero unique functionality.

Delete: `python-sdk/`, `python-examples/`, `python-models/`.

### `rpc` — modify endpoints

**Remove:** Model, target, submission query endpoints.

**Add:**
- `get_open_asks(filters?)` — sellers poll/subscribe to find asks
- `get_bids_for_ask(ask_id)` — buyer reviews bids
- `get_bid(bid_id)` — single bid lookup (needed for `soma accept` to infer ask_id)
- `get_settlements(address, role?)` — reputation data
- `get_reputation(address)` — computed summary
- `get_vault(address)` — seller vault balance
- `subscribe_asks(filters?)` — streaming subscription for new asks (websocket)

### `cli` — rewrite as primary interface

The CLI is the only client interface. It must be excellent — great defaults, clean abstractions, minimal required flags. An agent will use this; a human will use this. Design for both.

**Remove:** All model, submission, target commands. Remove `python-sdk` from workspace Cargo.toml members.

**Replace `send`, `pay`, `merge` with unified `transfer`:**
```
soma transfer <amount> <recipient>                    # simple: send SOMA to address
soma transfer <amount> <recipient> --usdc             # send USDC
soma transfer <amount> <r1> <r2> --amounts <a1>,<a2>  # multi-recipient
soma transfer --coins <c1>,<c2> <recipient>           # multi-coin input (auto-merge)
soma merge <coin1> <coin2> [<coin3>...]               # merge coins into first
```

**Marketplace commands with smart defaults:**
```
# Asks
soma ask create --task <file-or-stdin>                # hashes content → task_digest automatically
  --max-price <usdc>                                  # required
  --num-bids <n>                                      # default: 1
  --timeout <duration>                                # default: 5m, accepts human durations (30s, 5m, 1h, 1d)
soma ask cancel <ask-id>
soma ask list [--status open|filled|cancelled|expired] [--mine]
soma ask info <ask-id>

# Bids
soma bid create <ask-id> --price <usdc>
  --response <file-or-stdin>                          # hashes content → response_digest automatically
soma bid list --ask <ask-id>                          # list bids for an ask
soma bid list --mine                                  # list my bids across asks

# Accept — the core action. Smart defaults for common case.
soma accept <bid-id>                                  # infers ask-id from bid, auto-selects USDC coin
soma accept <bid-id> --payment-coin <ref>             # explicit coin override
soma accept --ask <ask-id> --cheapest                 # accept cheapest pending bid
soma accept --ask <ask-id> --cheapest --count <n>     # accept N cheapest bids

# Rating — only negative, only for sellers
soma rate <settlement-id>                             # submit negative seller rating
soma settlements [--as buyer|seller] [--address <addr>]

# Vault
soma vault                                            # show vault balance
soma vault withdraw [<amount>]                        # withdraw to wallet (default: all)

# Reputation
soma reputation [<address>]                           # default: own address
```

**Design principles:**
- Positional args for the common case, flags for overrides
- `--task` and `--response` accept file paths or stdin — CLI hashes the content into the digest. User never manually computes blake2b.
- `soma accept` infers the ask-id from the bid object (it's stored on-chain). No redundant `--ask-id` required.
- `soma accept --cheapest` is the power default for agents: "accept the best price for this ask."
- Durations accept human strings: `30s`, `5m`, `1h`, `1d`. Internally converted to milliseconds.
- USDC amounts accept decimal notation: `1.50` = 1,500,000 microdollars. SOMA amounts accept decimal: `1.5` = 1,500,000,000 shannons.
- JSON output mode (`--json`) for all commands, for agent consumption.

### `soma-graphql` — rewrite schema

**Delete:**
- `types/model.rs`
- `types/target.rs`
- `types/reward.rs`

**Add:**
- `types/ask.rs` — ask queries with filters (status, buyer, price range, time range)
- `types/bid.rs` — bid queries with filters (ask_id, seller, status)
- `types/settlement.rs` — settlement queries
- `types/reputation.rs` — reputation graph queries

**Reputation GraphQL schema:**

```graphql
type Reputation {
  address: SomaAddress!

  # As buyer (reputation derived from acceptance patterns, not explicit ratings)
  totalAsksCreated: Int!
  totalBidsAccepted: Int!
  totalVolumeSpent: BigInt!
  uniqueSellers: Int!                      # counterparty diversity — anti-sybil signal

  # As seller (reputation from explicit negative ratings + acceptance rate)
  totalBidsSubmitted: Int!
  totalBidsAccepted: Int!                  # "deliveries" — bids that reached settlement
  sellerApprovalRate: Float                # % of settlements without negative rating
  sellerApprovalRateRolling30d: Float
  bidToWinRatio: Float                     # bids accepted / bids submitted
  totalVolumeEarned: BigInt!

  # Graph data (for PageRank-like analysis)
  topCounterparties(limit: Int = 10): [CounterpartyEdge!]!
  ratingProvenance: RatingProvenance!
}

type CounterpartyEdge {
  address: SomaAddress!
  transactionCount: Int!
  totalVolume: BigInt!
  negativeRatings: Int!              # how many negative ratings in this pair
}

type RatingProvenance {
  fromEstablished: NegativeRatingStats!  # negative ratings from high-volume buyers (>N txns)
  fromNew: NegativeRatingStats!          # negative ratings from new/low-volume buyers
}

type NegativeRatingStats {
  totalSettlements: Int!
  negativeRatings: Int!
  negativeRate: Float                    # negativeRatings / totalSettlements
}
```

### `indexer-alt-schema` — rewrite soma tables

**Delete tables:**
- `soma_targets`
- `soma_models`
- `soma_target_reports`
- `soma_rewards`
- `soma_reward_balances`

**Modify tables:**
- `soma_epoch_state` — remove target/model fields (distance_threshold, hits_this_epoch, hits_ema, reward_per_target, targets_generated_this_epoch). Keep emission + safe_mode fields.
- `soma_staked_soma` — keep (validator staking only)
- `soma_validators` — keep as-is
- `soma_tx_details` — keep, update `kind` enum values

**New tables:**

```sql
CREATE TABLE soma_asks (
    ask_id BYTEA NOT NULL,
    cp_sequence_number BIGINT NOT NULL,
    buyer BYTEA NOT NULL,
    task_digest BYTEA NOT NULL,
    max_price_per_bid BIGINT NOT NULL,
    num_bids_wanted INTEGER NOT NULL,
    timeout_ms BIGINT NOT NULL,
    created_at_ms BIGINT NOT NULL,
    status TEXT NOT NULL,
    accepted_bid_count INTEGER NOT NULL,
    PRIMARY KEY (ask_id, cp_sequence_number)
);

CREATE TABLE soma_bids (
    bid_id BYTEA NOT NULL,
    cp_sequence_number BIGINT NOT NULL,
    ask_id BYTEA NOT NULL,
    seller BYTEA NOT NULL,
    price BIGINT NOT NULL,
    response_digest BYTEA NOT NULL,
    created_at_ms BIGINT NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (bid_id, cp_sequence_number)
);

CREATE TABLE soma_settlements (
    settlement_id BYTEA NOT NULL,
    cp_sequence_number BIGINT NOT NULL,
    ask_id BYTEA NOT NULL,
    bid_id BYTEA NOT NULL,
    buyer BYTEA NOT NULL,
    seller BYTEA NOT NULL,
    amount BIGINT NOT NULL,
    task_digest BYTEA NOT NULL,
    response_digest BYTEA NOT NULL,
    settled_at_ms BIGINT NOT NULL,
    seller_rating TEXT NOT NULL,           -- 'positive' (default) or 'negative'
    rating_deadline_ms BIGINT NOT NULL,
    PRIMARY KEY (settlement_id, cp_sequence_number)
);

CREATE TABLE soma_vaults (
    vault_id BYTEA NOT NULL,
    cp_sequence_number BIGINT NOT NULL,
    owner BYTEA NOT NULL,
    balance BIGINT NOT NULL,
    PRIMARY KEY (vault_id, cp_sequence_number)
);

-- Indexes for reputation graph queries
CREATE INDEX idx_settlements_buyer ON soma_settlements (buyer);
CREATE INDEX idx_settlements_seller ON soma_settlements (seller);
CREATE INDEX idx_settlements_task_digest ON soma_settlements (task_digest);
CREATE INDEX idx_settlements_settled_at ON soma_settlements (settled_at_ms);
CREATE INDEX idx_asks_buyer ON soma_asks (buyer);
CREATE INDEX idx_asks_status ON soma_asks (status);
CREATE INDEX idx_asks_created_at ON soma_asks (created_at_ms);
CREATE INDEX idx_bids_seller ON soma_bids (seller);
CREATE INDEX idx_bids_ask_id ON soma_bids (ask_id);
CREATE INDEX idx_bids_status ON soma_bids (status);
CREATE INDEX idx_vaults_owner ON soma_vaults (owner);
```

### `indexer-alt` — modify pipelines

Replace target/model/submission indexing pipelines with ask/bid/settlement/vault pipelines. The framework (`indexer-framework`) stays the same.

### Infrastructure crates — mostly unchanged

Unchanged:
- `consensus` — Mysticeti consensus
- `node` — validator/fullnode binary
- `store` — object storage
- `sync` — state sync
- `blobs` — blob storage
- `data-ingestion` — checkpoint ingestion
- `runtime` — async runtime
- `utils` — utilities (clean up any target/model failpoints)
- `indexer-framework` — generic pipeline architecture
- `indexer-pg-db` — Postgres backend
- `indexer-kvstore` — KV store backend
- `faucet` — testnet token distribution
- `test-cluster` — test infrastructure (update test scenarios)
- `e2e-tests` — rewrite test cases for marketplace flow + bridge flow

New crate:
- **`bridge-node`** — sidecar process per validator. Three subsystems: (1) Ethereum watcher using alloy crate — polls finalized blocks, queries `eth_getLogs` for deposit events; (2) gRPC signature exchange — peers share ECDSA signatures, submit certified actions when quorum reached; (3) Soma checkpoint watcher — observes `PendingWithdrawal` objects for withdrawal signing. Dependencies: `alloy`, `tonic` (gRPC), `secp256k1`.

New external (separate repo):
- **Ethereum bridge contract** — Solidity, holds USDC, deposit/withdraw/emergency/committee update

---

## Object Ownership and Contention Model

| Object | Owner | Contention | Who mutates |
|---|---|---|---|
| Ask | Shared (buyer + bidders need access) | Low — buyer mutates via AcceptBid/CancelAsk | Buyer |
| Bid | Shared (buyer accepts) | Very low — created once, mutated once (accept or expire) | Buyer (accept) |
| Settlement | Shared (buyer may rate) | Very low — at most 1 rating mutation | Buyer (rate seller negative) |
| SellerVault | Owned by seller | None between sellers — only seller writes | System (AcceptBid credits), Seller (withdraw debits) |
| PendingWithdrawal | System (read-only after creation) | None — created once, read by bridge nodes | System (created on BridgeWithdraw) |

Note: Asks and Bids must be shared objects because multiple parties interact with them. Settlements must be shared because the buyer may rate.

SellerVault has self-contention: if a buyer accepts two bids from the same seller simultaneously, both AcceptBid transactions try to credit the same vault. Mysticeti's consensus ordering handles this correctly (they execute sequentially). In practice, concurrent accepts to the same seller are rare.

---

## Workspace Cargo.toml Changes

**Remove from members:**
```toml
"scoring",
"models",
"arrgen",        # verify no other crate depends on this
"python-sdk",    # CLI is the only client interface; Python wrapper adds build complexity for zero unique functionality
```

**Keep all other members.**

---

## Off-Chain Components (separate repos, for reference)

### Glass Server

SaaS service that abstracts the on-chain marketplace for agent consumers:
- Accepts credit card payments for credits (off-chain billing — consumer never touches crypto)
- Receives full task content from consumers (chain only sees the hash)
- Spends its own on-chain USDC to create asks, accept bids on behalf of consumers
- Broadcasts asks to sellers (websocket with optional keyword/tag filters)
- Relays responses from sellers to consumers
- Exposes REST + WebSocket API

Interacts with chain via RPC: creating asks/bids, accepting, delivering, rating. Glass is the on-chain actor — consumers are off-chain clients of Glass.

### MCP Server

Wraps the Glass server as MCP tools for any agent:
- `delegate(task, max_price, num_competitors, timeout)` — posts ask, waits, returns best response
- `get_reputation(address)` — queries chain
- `get_market_stats()` — live ask/bid activity

### Agent Framework (from PoC concepts)

Client-side, not protocol-level. Agents use the CLI (`soma` binary) as their interface — JSON output mode (`--json`) for structured parsing, human-readable defaults for interactive use.

- Monitors asks via `soma ask list --status open --json` or RPC/websocket subscription
- Local routing: decides which asks to bid on based on agent's skill/context
- Bids via `soma bid create <ask-id> --price <usdc> --response <file>`
- Tracks local results (win rate by task type, approval history via `soma reputation --json`)
- Reflection loop: adjusts bidding strategy and solver based on outcomes

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

- `deposit`: user sends USDC to the contract, specifying their Soma address. Emits `Deposited(nonce, sender, somaRecipient, amount)` event. Nonce is auto-incremented.
- `withdraw`: verifies aggregated ECDSA signatures from validators representing sufficient stake, then releases USDC to recipient. Nonce prevents replay.
- `updateCommittee`: called at each epoch boundary to sync the bridge committee with the current validator set. Requires 2/3 stake from the *current* committee.

#### 2. Soma on-chain (new types and transactions)

##### Bridge state in SystemState

```rust
pub struct BridgeState {
    pub paused: bool,
    pub next_withdrawal_nonce: u64,
    pub processed_deposit_nonces: BTreeSet<u64>,  // prevent replay
    pub bridge_committee: BridgeCommittee,
    pub total_bridged_usdc: u64,                  // total USDC minted via bridge
}

pub struct BridgeCommittee {
    pub members: BTreeMap<SomaAddress, BridgeMember>,
    pub threshold_deposit: u64,     // f+1, ~3334/10000
    pub threshold_withdraw: u64,    // f+1, ~3334/10000
    pub threshold_pause: u64,       // ~450/10000
    pub threshold_unpause: u64,     // 2/3, ~6667/10000
}

pub struct BridgeMember {
    pub ecdsa_pubkey: [u8; 33],     // compressed Secp256k1 public key
    pub voting_power: u64,
}
```

##### New transaction types

```rust
// === System transactions (gasless, like ConsensusCommitPrologueV1) ===

/// Certified deposit — submitted once off-chain signature collection reaches quorum.
/// Mints CoinType::Usdc to recipient. Gasless system tx — bridge nodes submit without paying gas.
BridgeDeposit(BridgeDepositArgs),

/// Emergency pause — stops all bridge operations. Gasless system tx.
BridgeEmergencyPause(BridgeEmergencyPauseArgs),

/// Emergency unpause — resumes bridge operations. Gasless system tx.
BridgeEmergencyUnpause(BridgeEmergencyUnpauseArgs),

// === User transactions (normal gas) ===

/// User initiates withdrawal — burns their USDC, creates a pending withdrawal
/// that bridge nodes will sign off-chain for Ethereum release.
BridgeWithdraw(BridgeWithdrawArgs),
```

```rust
pub struct BridgeDepositArgs {
    pub nonce: u64,
    pub eth_tx_hash: [u8; 32],
    pub recipient: SomaAddress,
    pub amount: u64,
    pub aggregated_signature: Vec<u8>,  // aggregated ECDSA sigs from bridge committee
    pub signer_bitmap: Vec<u8>,        // which committee members signed
}

pub struct BridgeWithdrawArgs {
    pub payment_coin: ObjectRef,       // CoinType::Usdc to burn
    pub amount: u64,
    pub recipient_eth_address: [u8; 20],
}

pub struct BridgeEmergencyPauseArgs {
    pub aggregated_signature: Vec<u8>,
    pub signer_bitmap: Vec<u8>,
}

pub struct BridgeEmergencyUnpauseArgs {
    pub aggregated_signature: Vec<u8>,
    pub signer_bitmap: Vec<u8>,
}
```

##### Execution logic

**BridgeDeposit:**
1. Validate bridge is not paused
2. Validate nonce not in `processed_deposit_nonces`
3. Verify aggregated signature against bridge committee — total signing stake >= `threshold_deposit`
4. Mint new `CoinType::Usdc` coin with `amount` to `recipient`
5. Add nonce to `processed_deposit_nonces`
6. Increment `total_bridged_usdc`

**BridgeWithdraw:**
1. Validate bridge is not paused
2. Validate `payment_coin` is `CoinType::Usdc`
3. Burn the coin (delete object, deduct from `total_bridged_usdc`)
4. Create `PendingWithdrawal` object with `next_withdrawal_nonce`, increment nonce
5. Bridge nodes observe this in checkpoints and begin off-chain signing for Ethereum release

```rust
pub struct PendingWithdrawal {
    pub id: ObjectID,
    pub nonce: u64,
    pub sender: SomaAddress,
    pub recipient_eth_address: [u8; 20],
    pub amount: u64,
    pub created_at_ms: u64,
}
```

**BridgeEmergencyPause:** verify signatures meet `threshold_pause`, set `paused = true`.

**BridgeEmergencyUnpause:** verify signatures meet `threshold_unpause`, set `paused = false`.

#### 3. Bridge node (off-chain, runs per validator)

Each validator runs a bridge node as a sidecar process. Three subsystems:

##### Ethereum watcher (modeled after Sui's `EthSyncer` + `EthClient`)

Uses the [alloy](https://github.com/alloy-rs/alloy) crate for Ethereum RPC. Two concurrent tasks:

**Finalized block poller** — every ~5 seconds, calls `eth_getBlockByNumber("finalized", false)` to get the latest finalized Ethereum block. Only finalized blocks are processed (no reorg risk). Broadcasts new block numbers to the event listener via a `watch` channel.

**Event listener** — waits for new finalized blocks, then queries `eth_getLogs` with a `Filter` on the bridge contract address and block range. Queries in chunks (max ~1000 blocks per call). Handles RPC `-32005` errors (too many results) by halving the query window and retrying. Emits parsed deposit events for the signature exchange subsystem.

```rust
struct EthWatcher {
    provider: alloy::providers::RootProvider,  // standard Ethereum JSON-RPC
    bridge_contract: [u8; 20],                 // Soma bridge contract on Ethereum
    last_processed_block: u64,
}
```

Key Ethereum RPC calls:
- `eth_getBlockByNumber("finalized", false)` — latest finalized block
- `eth_getLogs({ address, fromBlock, toBlock })` — deposit events from bridge contract
- `eth_getTransactionReceipt(tx_hash)` — verify event provenance when needed

Configure with multiple RPC endpoints (Infura, Alchemy, self-hosted) with fallback. RPC failures beyond a threshold trigger automatic bridge pause submission.

##### gRPC signature exchange

Bridge nodes discover peers via the validator set (network addresses from `ValidatorInfo`). When a deposit event or pending withdrawal is observed:

1. Sign the message with the validator's ECDSA bridge key
2. Broadcast signature to all peer bridge nodes via gRPC
3. Collect incoming signatures from peers
4. When signatures representing sufficient stake are collected, any node can submit the certified action

```protobuf
service BridgeNode {
    rpc SubmitSignature(SignatureRequest) returns (SignatureResponse);
}

message SignatureRequest {
    bytes action_digest = 1;   // keccak256 of the bridge action
    bytes signature = 2;       // ECDSA signature
    uint32 signer_index = 3;   // index in bridge committee
}
```

##### Checkpoint watcher (Soma side)

Watches Soma checkpoints for `PendingWithdrawal` objects. For each new withdrawal:
1. Construct the withdrawal release message
2. Feed it into the gRPC signature exchange
3. When quorum reached, submit `withdraw()` call to the Ethereum contract

**Deposit flow (Ethereum → Soma):**
1. Ethereum watcher observes `Deposited` event in finalized block
2. Signs deposit message, exchanges sigs with peers via gRPC
3. When f+1 stake collected, submits gasless `BridgeDeposit` system tx to Soma
4. Soma mints CoinType::Usdc to recipient

**Withdrawal flow (Soma → Ethereum):**
1. Checkpoint watcher observes `PendingWithdrawal` on Soma
2. Signs withdrawal release message, exchanges sigs with peers via gRPC
3. When f+1 stake collected, submits `withdraw()` to Ethereum bridge contract
4. Ethereum contract verifies signatures and releases USDC to recipient

**Committee sync:**
- At each epoch boundary, bridge nodes observe the new validator set
- Sign a committee update message
- Submit `updateCommittee()` to the Ethereum contract when quorum reached

### Validator key management

Each validator now holds three keypairs:
- **BLS12-381**: consensus protocol key (existing)
- **Ed25519**: network/worker key (existing)
- **Secp256k1**: bridge signing key (new) — registered in genesis and updateable via validator metadata

The bridge key is included in `ValidatorInfo` and the genesis config. Validators can rotate their bridge key at epoch boundaries via `UpdateValidatorMetadata`.

### Security considerations

- **Ethereum RPC dependency**: bridge nodes depend on Ethereum RPC for deposit observation. RPC failures should trigger automatic pause (like Hyperliquid's October 2024 incident). Multiple RPC endpoints with fallback.
- **Nonce sequencing**: deposits use Ethereum contract nonces, withdrawals use Soma-side nonces. Both prevent replay.
- **Pause asymmetry**: pausing is fast and cheap (~5% stake), unpausing requires strong consensus (2/3). Safety over liveness.
- **No withdrawal dispute window in v1**: Sui has a more complex dispute mechanism. For v1, the signature quorum IS the security — if f+1 validators sign a withdrawal, it's valid. Dispute windows can be added later.

### Future bridge upgrades (not in this refactor)

- **Governance actions**: blocklisting bridge members, transfer limits, rate limiting
- **Asset price oracles**: for limit enforcement based on USD value
- **Additional token support**: bridging SOMA or other assets
- **Dispute windows**: time-delayed withdrawals with challenge period
- **Multi-chain**: bridging from chains beyond Ethereum

---

## Protocol Version and Deployment

This is a breaking protocol change. Requires a new protocol version and a **fresh testnet** — the state model is too different for a live migration to be worth engineering.

### Testnet (this refactor)

- Bump protocol version in `protocol-config`
- New genesis config with marketplace parameters, bridge committee (validator ECDSA keys), SOMA allocation to validators
- No USDC at genesis — all USDC enters via the bridge (Glass bridges its own USDC from Ethereum like any other user)
- USDC bridge live from day one — deploy Ethereum contract, validators run bridge nodes
- Glass service as SaaS on-ramp (credit card → credits → Glass spends its own on-chain USDC)
- Sellers and advanced users can also bridge USDC directly via Ethereum (permissionless)
- `value_fee_bps` active from day one — marketplace fees accumulate in Protocol Fund
- Sellers earn real USDC in SellerVaults, can withdraw off-chain via bridge
- SOMA faucet distributes gas tokens for transaction fees
- All validators upgrade simultaneously (coordinated via epoch change, or fresh start)
- Update K8s manifests (validators + bridge nodes + indexer + graphql)
- Settlement graph data accumulates for retroactive distribution

### Mainnet (future protocol upgrades, in order)

1. **On-chain DEX** — AMM or order book for SOMA/USDC pair. Enables price discovery and the buyback mechanism.
2. **Buyback-and-burn** — Protocol Fund continuously purchases SOMA via the DEX and burns it. Activated by governance/protocol upgrade once DEX liquidity is sufficient.
3. **Retroactive airdrop** — SOMA distributed to testnet participants weighted by settlement graph metrics (volume, counterparty diversity, approval quality, longevity).
4. **Fee tiers** — volume-based discounts, SOMA staking for additional discounts. Layered on once marketplace has proven sustained activity.
5. **Bridge governance** — blocklisting, transfer limits, rate limiting, dispute windows.

---

## Migration Sequencing

### Progress Summary (updated 2026-03-29, session 6)

**Phase 1: COMPLETE.** All 12 items done.

**Phase 2: COMPLETE.** All 14 items done. The workspace compiles cleanly under both normal and msim builds (`cargo check` passes with zero errors). All 234 types crate unit tests pass.

**Phase 3: COMPLETE.** All 11 items done. Marketplace executor, bridge executor, dispatch wiring, and unit tests are all implemented and passing. 29 new tests (21 marketplace + 8 bridge). See details below.

**Phase 4: COMPLETE.** All 6 items done. Fresh-testnet approach — deleted old migrations and rewrote from scratch (no ALTER TABLE). See details below.

**Phase 5: MOSTLY COMPLETE.** SDK cleanup, CLI marketplace commands, RPC cleanup, unified transfer, accept auto-USDC, marketplace RPC query endpoints (GetAsk, GetBidsForAsk, GetOpenAsks, GetSettlement, GetSettlements, GetVault, GetProtocolFund, GetReputation, SubscribeAsks/SubscribeBids), accept --cheapest wired to live bids_by_ask index, seller/buyer listen CLI commands, `soma settlements` and `soma reputation` CLI commands, and settlements secondary indexes (by buyer and seller) all done. RPC `ObjectType` updated to support `Coin(SOMA)`/`Coin(USDC)` and all marketplace/bridge object types. Proto definitions and full round-trip conversions added for all marketplace and bridge transaction kinds. SDK `get_reputation()` and `get_protocol_fund()` methods added. CLI `soma reputation` now uses server-side GetReputation RPC instead of client-side computation. See details below.

**Phase 6: IN PROGRESS.** 32 e2e tests written and passing (26 marketplace + 6 bridge, all with real ECDSA). Full e2e suite passes with 0 failures. USDC genesis support added to TestClusterBuilder. Unified Transfer tests (SOMA single, USDC single, multi-recipient), MergeCoins tests (SOMA, USDC), CoinType enforcement tests (mixed Transfer rejected, mixed MergeCoins rejected), ask expiry test, USDC+epoch supply conservation test, default positive rating (deadline expiry), vault accumulation (multi-ask), and secondary index consistency (bids_by_ask + open_asks) all done. **Bridge e2e tests added** — `TestClusterBuilder::with_bridge_committee(n)` generates an n-member committee with equal voting power (total 10000). Bridge committee threaded through `GenesisConfig` → `GenesisBuilder` → `SystemState::create()`. 6 tests cover: BridgeDeposit minting USDC, nonce replay rejection, BridgeWithdraw (burn + PendingWithdrawal), emergency pause/unpause flow (pause blocks ops, unpause resumes), insufficient stake rejection, and full deposit→withdraw→deposit round-trip. **USDC supply conservation bug fixed** — `check_soma_conservation()` now only sums `CoinType::Soma` coins, excluding USDC from the SOMA supply check. **RPC field mask bug fixed** — all marketplace RPC endpoints now include `previous_transaction` in their read masks. **TestClusterBuilder now supports `with_marketplace_params()` and `with_bridge_committee()`** for custom genesis parameters. See details below.

**Phase 3b: IN PROGRESS.** ECDSA verification in bridge executor is done. `bridge-node` crate now has 10 modules and **34 unit tests passing** (up from 15). Retry with exponential backoff added. EthClient and EthSyncer have wiremock-based integration tests. Crypto cross-verification tests cover sign→ecrecover roundtrip for all action types with known test vectors for future Solidity cross-verification. Ethereum contract (Solidity) not started. See details below.

Key findings from Sui analysis:
- Sui's `EthClient` and `EthSyncer` patterns are directly reusable with minor simplification (USDC-only instead of multi-token).
- Sui's `BridgeCommittee.sol` signature verification (`ecrecover` + bitmap stake summation) maps cleanly to our `verify_committee_stake()` which currently defers real ECDSA to Phase 3b.
- fastcrypto's `Secp256k1KeyPair` with `sign_recoverable_with_hash::<Keccak256>` is the Rust-side signing pattern (already in our workspace deps).
- Message encoding (`MESSAGE_PREFIX + type + version + nonce + chainID + payload`, keccak256) must match between Solidity and Rust for cross-chain signature verification.
- Our bridge is substantially simpler than Sui's: no multi-token (BridgeConfig, BridgeVault, decimal conversion), no rate limiting (BridgeLimiter), no blocklisting, no Move event watching (we use checkpoint object scanning).
- Ethereum contract: 3 Solidity files (SomaBridgeMessage, SomaBridgeCommittee, SomaBridge) vs Sui's 10+. Foundry tests.
- Bridge node: new `bridge-node` crate with 9 modules. ~2500 lines estimated. Dependencies: alloy, tonic, fastcrypto, types.

**What Phase 3 did so far:**
- Created `authority/src/execution/marketplace.rs` — full `MarketplaceExecutor` implementing all 6 marketplace transaction types:
  - `CreateAsk`: validates timeout bounds against `MarketplaceParameters`, price > 0, num_bids > 0. Creates Ask object as shared (buyers + bidders interact). Uses `epoch_start_timestamp_ms` as `created_at_ms`.
  - `CancelAsk`: validates sender == buyer, ask is Open, no bids accepted yet. Sets status Cancelled.
  - `CreateBid`: validates ask Open, within timeout, price <= max, price > 0, seller != buyer. Creates Bid as shared object with response_digest.
  - `AcceptBid`: atomic settlement — validates bid Pending, ask not full, payment coin is USDC. Computes `marketplace_fee = bps_mul(bid.price, marketplace_fee_bps)`. Deducts `bid.price` from buyer's USDC coin. Credits fee to `protocol_fund_balance`. Creates new `SellerVault` per settlement (avoids contention on single vault). Creates Settlement with `seller_rating: Positive` and `rating_deadline_ms`. Increments `accepted_bid_count`, sets ask Filled when full.
  - `RateSeller`: validates sender == buyer, rating currently Positive, within deadline. Sets `seller_rating = Negative`.
  - `WithdrawFromVault`: validates sender == vault.owner. Deducts amount (or full balance). Credits to existing USDC coin or creates new one. Deletes vault if balance == 0.
- Created `authority/src/execution/bridge.rs` — full `BridgeExecutor` implementing all 4 bridge transaction types, modeled after Sui's native bridge patterns:
  - `BridgeDeposit`: validates not paused, nonce not replayed (mirrors Sui's `sequence_nums` / EVM `isTransferProcessed`), verifies committee stake threshold via signer_bitmap, mints `CoinType::Usdc` to recipient, records nonce.
  - `BridgeWithdraw`: validates not paused, payment coin is USDC. Burns USDC (deducts or deletes). Creates `PendingWithdrawal` as immutable object (bridge nodes observe via checkpoints). Increments withdrawal nonce.
  - `BridgeEmergencyPause`: verifies low-threshold (~5% stake) committee stake, sets `paused = true`.
  - `BridgeEmergencyUnpause`: verifies high-threshold (2/3 stake) committee stake, sets `paused = false`.
  - Committee stake verification follows Sui's pattern: iterate BTreeMap members in order, check bitmap bits, sum voting_power, reject if < threshold. Full ECDSA ecrecover deferred to Phase 3b when secp256k1 crate is integrated.
- Wired both executors into `create_executor()` dispatch in `execution/mod.rs` — replaced `todo!()` placeholders.
- Added 18 new `ExecutionFailureStatus` error variants: marketplace errors (AskNotFound, AskNotOpen, AskExpired, AskAlreadyFilled, AskHasAcceptedBids, BidNotFound, BidNotPending, BidPriceTooHigh, SellerCannotBidOnOwnAsk, SettlementNotFound, SettlementAlreadyRatedNegative, RatingDeadlinePassed, VaultNotFound, InsufficientVaultBalance, WrongCoinTypeForPayment) and bridge errors (BridgePaused, BridgeNonceAlreadyProcessed, BridgeInsufficientSignatureStake).
- Updated RPC match arms (`rpc_proto_conversions.rs`, `types_conversions.rs`) for all new error variants.
- Added `SystemState` accessor methods: `marketplace_params()`, `protocol_fund_balance()`, `add_protocol_fund_balance()`, `bridge_state()`, `bridge_state_mut()`.
- Added `Object::new_marketplace_object()` generic constructor for BCS-serialized marketplace/bridge objects, `Object::deserialize_contents()` for typed deserialization, and `Object::update_contents()` for BCS re-serialization.
- Fixed pre-existing test issues from Phase 1/2: broken `ObjectRef` tuple syntax in `transaction_validation_tests.rs`, removed `target_state()` tests in `epoch_tests.rs` (target system was stripped), updated `emission_per_epoch` reference to `current_distribution_amount`.
- Workspace compiles cleanly: `cargo check` (normal) and `RUSTFLAGS="--cfg msim" cargo check` both pass with zero errors. 234 types crate tests pass. 182 authority crate tests pass (0 failures).
- **Unit tests for marketplace executor** (21 tests in `authority/src/unit_tests/marketplace_tests.rs`):
  - Happy paths: CreateAsk, CreateBid, CancelAsk, AcceptBid (full flow: ask→bid→accept, verifies settlement creation, vault credited, bid accepted, ask filled, USDC deducted, marketplace fee computed), RateSeller (negative), WithdrawFromVault (full balance with vault deletion, partial balance).
  - Edge cases: zero price rejected, zero bids rejected, timeout too short rejected, bid price too high rejected, seller bids on own ask rejected, wrong coin type (SOMA instead of USDC) rejected, insufficient balance rejected, over-accept rejected (ask already filled), double rate rejected (already negative), wrong sender for cancel/rate rejected, wrong vault owner rejected, insufficient vault balance rejected.
- **Unit tests for bridge executor** (8 tests in `authority/src/unit_tests/bridge_tests.rs`):
  - BridgeDeposit: insufficient stake rejected (empty committee), nonce replay detection (verifies reverted execution doesn't record nonce).
  - BridgeWithdraw: burns USDC and creates PendingWithdrawal (verifies balance deduction, object creation, sender/recipient/amount), exact-amount withdraw deletes coin, insufficient balance rejected, wrong coin type (SOMA) rejected.
  - BridgeEmergencyPause/Unpause: insufficient stake rejected.
- **Bug fixes discovered during testing:**
  1. **Shared object input registration (critical).** Marketplace transactions (CancelAsk, CreateBid, AcceptBid, RateSeller) were not listing their shared objects (Ask, Bid, Settlement) in `input_objects()` or `shared_input_objects()` in `types/src/transaction.rs`. The executor would read them via `store.read_object()` but they were never loaded into the TemporaryStore. Fixed by adding `InputObjectKind::SharedObject` entries for each marketplace-specific shared object.
  2. **Shared object initial_shared_version.** Marketplace shared objects (Ask, Bid, Settlement) were created with `Owner::Shared { initial_shared_version: Version::MIN }` (Version(0)), but the TemporaryStore's `update_version_and_previous_tx()` rewrites `initial_shared_version` to the lamport timestamp for objects created with `Version::new()` (=0). This made the version unpredictable. Fixed by using `OBJECT_START_VERSION` (Version(1)), which the TemporaryStore preserves. This matches the pattern used by the old Target shared objects.
  3. **Shared object mutability for version tracking.** CreateBid initially listed the Ask as `mutable: false` (read-only), but `assign_versions_for_certificate()` advances the `next_shared_object_versions` counter for ALL shared inputs regardless of mutability. When the ask wasn't written back (immutable), subsequent transactions got assigned a version that didn't exist in the store. Fixed by marking all shared marketplace objects as `mutable: true` so `ensure_active_inputs_mutated()` bumps their version even when the executor only reads them.
  4. **Bridge total_bridged_usdc underflow.** `BridgeWithdraw` used `checked_sub` for `total_bridged_usdc` tracking, but USDC can exist on-chain without a prior `BridgeDeposit` (e.g., genesis allocation, testing). Fixed by using `saturating_sub` since `total_bridged_usdc` is a best-effort tracking counter, not a security check.
  5. **Transfer coin fee check with separate gas (pre-existing).** `authority/src/execution/coin.rs` checked `total_available < total_payments + remaining_fee` even when the gas coin was separate from the transfer coins. This caused `InsufficientCoinBalance` when transferring the full balance of a non-gas coin because fees should come from the gas coin, not the transfer coins. Fixed by only including `remaining_fee` in the check when the gas coin is among the transfer coins (`has_gas_coin`). This fixes `test_transfer_coin_full_amount_non_gas`.

**Design decision: SellerVault per settlement.** The REFACTOR.md spec says "load or create seller's SellerVault" in AcceptBid, but the vault is an owned object (owned by seller). The buyer's AcceptBid transaction cannot load the seller's owned objects as inputs. Solution: AcceptBid creates a new SellerVault per settlement. The seller uses WithdrawFromVault to drain individual vaults into USDC coins. This avoids cross-owner contention and keeps AcceptBid's input set minimal (system_state + ask + bid + payment_coin). Future optimization: a system-level vault registry in SystemState could aggregate, but per-settlement vaults work correctly and avoid contention.

**Design decision: shared object mutability.** All dynamically-created shared objects (Ask, Bid, Settlement) MUST be listed as `mutable: true` in every transaction that references them, even read-only ones. This is because the consensus version manager increments the next-version counter for all shared inputs, and if the object's version in the store doesn't advance (because it wasn't written), subsequent transactions will get assigned a version that doesn't exist. The `ensure_active_inputs_mutated()` function handles auto-bumping the version for mutable inputs that weren't explicitly written. Use `OBJECT_START_VERSION` (not `Version::MIN`) for `initial_shared_version` so the TemporaryStore preserves the version predictably.

**What Phase 4 did:**
- **Fresh-testnet migration approach.** Since testnet restarts from scratch, deleted all old target/model/reward migration directories (9 directories: `soma_targets`, `soma_models`, `soma_rewards`, `soma_targets_denormalize`, `soma_models_denormalize`, `soma_reward_balances`, `soma_target_models`, `soma_target_reports`, `soma_models_pending_update`). Rewrote `soma_epoch_state` migration in place with new schema (no ALTER TABLE). Created 4 clean new migrations for marketplace tables.
- **Rewrote `indexer-alt-schema/src/schema.rs`.** Removed 5 old `diesel::table!` definitions (`soma_targets`, `soma_models`, `soma_rewards`, `soma_reward_balances`, `soma_target_reports`). Added 4 new tables (`soma_asks`, `soma_bids`, `soma_settlements`, `soma_vaults`). Updated `soma_epoch_state` — removed `distance_threshold`, `targets_generated_this_epoch`, `hits_this_epoch`, `hits_ema`, `reward_per_target`; added `distribution_counter`, `period_length`, `decrease_rate`, `protocol_fund_balance`. Updated `allow_tables_to_appear_in_same_query!`.
- **Rewrote `indexer-alt-schema/src/soma.rs`.** Removed `StoredTarget`, `StoredModel`, `StoredTargetReport`, `StoredReward`, `StoredRewardBalance`. Added `StoredAsk`, `StoredBid`, `StoredSettlement`, `StoredVault`. Updated `StoredEpochState` to match new schema.
- **Created 4 new indexer-alt handlers:** `soma_asks.rs`, `soma_bids.rs`, `soma_settlements.rs`, `soma_vaults.rs`. Each scans checkpoint output objects for the corresponding `ObjectType`, deserializes via BCS (`Object::deserialize_contents()`), and produces `Stored*` rows. Registered all 4 as Tier C (never pruned) concurrent pipelines in `indexer-alt/src/lib.rs`.
- **Updated `soma_epoch_state` handler.** Reads `protocol_fund_balance`, `distribution_counter`, `period_length`, `decrease_rate` from `SystemStateV1`. Removed target/model field zeroing.
- **Rewrote GraphQL types.** Deleted `types/reward.rs` and `types/aggregates.rs`. Created `types/ask.rs`, `types/bid.rs`, `types/settlement.rs`, `types/reputation.rs`. Updated `types/epoch_state.rs` — replaced target/model fields with `distribution_counter`, `period_length`, `decrease_rate`, `protocol_fund_balance`.
- **Rewrote GraphQL query resolvers.** Removed `rewards()` and `reward_aggregates()` queries. Added `asks(status?, buyer?, limit?)`, `bids(ask_id?, seller?, status?, limit?)`, `settlements(buyer?, seller?, limit?)`, `reputation(address)` queries. Updated `epoch_state()` query to use new schema. Reputation query computes seller approval rate, bid-to-win ratio, unique sellers, and volume from `soma_settlements` and `soma_bids` tables via SQL aggregation.
- **Removed old DataLoaders.** Deleted `TargetReportersLoader` and `TargetRewardLoader` from `soma-graphql/src/loaders.rs`. Removed `DataLoader` registration from `build_schema()`.
- **Updated GraphQL tests.** Replaced `test_rewards_by_epoch`, `test_rewards_filter_by_target`, `test_rewards_empty`, `test_reward_aggregates` with `test_asks_query`, `test_settlements_query`, `test_reputation_query`. Updated `test_epoch_state_query` for new fields.
- **Workspace compiles cleanly:** `cargo check` (normal) and `RUSTFLAGS="--cfg msim" cargo check` both pass. All types (234), authority (182), indexer-alt (29 ignored — require Postgres), and graphql tests compile and pass (graphql_tests are ignored — require Postgres).

**What Phase 5 did so far:**
- **SDK cleanup.** Removed `list_targets()`, `get_architecture_version()`, `score()`, `scoring_health()`, `scoring_url()` builder method, and all `scoring` type aliases/imports/fields. The `grpc-services` feature now only gates `admin`. Added 6 marketplace transaction builder methods: `create_ask()`, `cancel_ask()`, `create_bid()`, `accept_bid()`, `rate_seller()`, `withdraw_from_vault()`. Each builds the `TransactionKind`, auto-selects gas, signs, and executes. SDK compiles cleanly (24 tests pass).
- **CLI marketplace commands.** Created 5 new command modules: `commands/ask.rs` (create with `--task` file auto-hashing, `--max-price` USDC, `--timeout` human duration, `--num-bids`; cancel), `commands/bid.rs` (create with `--response` file auto-hashing, `--price` USDC), `commands/accept.rs` (accepts bid by ID, infers ask_id from bid object, requires `--payment-coin`), `commands/rate.rs` (negative seller rating on settlement), `commands/vault.rs` (withdraw USDC from seller vault with optional amount). Added `usdc_amount.rs` with `UsdcAmount` type (6 decimal places, microdollars) and `parse_duration_ms()` helper (accepts `30s`, `5m`, `1h`, `1d`, `500ms`). Registered all 6 top-level commands in `SomaCommand` enum (`ask`, `bid`, `accept`, `rate`, `vault`) with full clap help text and wired dispatch in `execute()`. Removed 3 dead model tests from `cli_integration_tests.rs`. CLI compiles cleanly (25 lib tests + 15 integration tests pass).
- **RPC cleanup.** Removed old target/challenge gRPC endpoints from `state_service.proto` (GetTarget, ListTargets, GetChallenge, ListChallenges). Removed corresponding stub implementations from `state_service/mod.rs`. Added `GetBid` RPC endpoint — looks up a bid object by ID using the existing object store, returns it with standard field masking. Proto regenerates cleanly. Removed `list_targets()` and `get_architecture_version()` from `rpc/src/api/client.rs`. RPC compiles cleanly.
- **Workspace compiles cleanly:** `cargo check` (normal) and `RUSTFLAGS="--cfg msim" cargo check` both pass. All types (234), authority (182), cli (25 lib + 15 integration), sdk (24) tests pass.

**What Phase 5 continued (CLI redesign for agent ergonomics):**
- **Unified `soma transfer`.** Replaced `send`/`pay` with `soma transfer <amount> <recipient> [--usdc]`. Positional args for the common case — agents just do `soma transfer 1.50 0xABC --usdc`. Multi-recipient: `soma transfer 0 0xA 0xB --amounts 3,2`. Old `send`/`pay` hidden but still work. Object transfer moved to `transfer-object`. `merge-coins` aliased as `merge`.
- **Accept auto-USDC.** `soma accept <BID_ID>` auto-selects richest USDC coin — agents no longer need to find their USDC coin object ID. Added `WalletContext::get_richest_usdc_coin()` and `get_usdc_coins_sorted_by_balance()` which fetch all coins and filter by `CoinType::Usdc` client-side (RPC's ObjectType enum doesn't discriminate CoinType yet).
- **Accept `--cheapest` stub.** `soma accept --ask 0xASK --cheapest [--count N]` is wired up but blocked on `get_bids_for_ask` RPC endpoint. Returns helpful error directing user to `soma bid list --ask ...` then `soma accept <BID>`. When the RPC endpoint lands, `execute_cheapest()` just needs the bid listing call.
- **Design decisions for agent UX:**
  - USDC is the default for marketplace commands (agents use USDC). SOMA is the default for `transfer` (gas token). `--usdc` flag switches.
  - Auto-selection everywhere: gas coin, payment coin, ask_id inference from bid. Agents should never need to hunt for object IDs unless they want explicit control.
  - `--json` on every command. Agents parse structured output.
  - Minimal required flags. `soma accept <BID>` is one positional arg. `soma ask create --task file --max-price 1 --timeout 5m` has smart defaults (`--num-bids 1`).
  - `--cheapest` on accept is the agent power-move: "accept the best deal for my ask." Now fully wired to live bids_by_ask RPC index.
- **Workspace compiles cleanly:** `cargo check` (normal) and `RUSTFLAGS="--cfg msim" cargo check` both pass. All types (234), authority (182), cli (25 lib + 15 integration), sdk (24) tests pass.

**What Phase 5 continued (marketplace RPC endpoints + listen UX):**
- **RPC secondary indexes.** Added two new RocksDB tables to `IndexStoreTables` in `authority/src/rpc_index.rs`: `bids_by_ask: DBMap<(ObjectID, ObjectID), ()>` (index of bids by ask ID for efficient lookup) and `open_asks: DBMap<ObjectID, ()>` (index of currently-open asks for marketplace discovery). Both indexes are populated during checkpoint indexing (`index_objects`) and real-time tx indexing (`index_executed_tx_objects`). Bid entries are added when Bid objects are created (deserializes BCS to extract `ask_id`), removed when deleted. Ask entries track open status — added when Ask is Open, removed when Filled/Cancelled/Expired. Bumped `CURRENT_DB_VERSION` to 5.
- **Extended `RpcIndexes` trait.** Added `bids_for_ask(&ObjectID) -> Vec<ObjectID>` and `open_asks() -> Vec<ObjectID>` to `types/src/storage/read_store.rs`. Implemented on `RestReadStore` in `authority/src/storage.rs` delegating to the new RocksDB tables.
- **Marketplace RPC endpoints.** Added 7 new RPCs to `state_service.proto`:
  - `GetAsk(ask_id)` — direct object lookup by ID (same pattern as GetBid).
  - `GetSettlement(settlement_id)` — direct object lookup by ID.
  - `GetBidsForAsk(ask_id, status_filter?)` — uses `bids_by_ask` secondary index to find all bid IDs, fetches each bid object, optional Pending/Accepted/etc. status filter via BCS deserialization.
  - `GetOpenAsks(buyer?, page_size?)` — uses `open_asks` secondary index, optional buyer address filter.
  - `GetVault(owner)` — uses existing `owned_objects_iter` with `ObjectType::SellerVault` filter (vaults are owned objects).
  - `SubscribeAsks(min_max_price?)` — server-streaming RPC for real-time ask events (stub returning empty stream — broadcast channel wiring deferred to Phase 6 when checkpoint processing hooks are available).
  - `SubscribeBids(ask_id)` — server-streaming RPC for real-time bid events on a specific ask (stub returning empty stream — same deferral).
- **Refactored state_service handler.** Extracted `lookup_object_by_id()` helper to deduplicate the pattern of ID parse → object store lookup → field mask → proto conversion (used by GetBid, GetAsk, GetSettlement).
- **RPC client methods.** Added to `rpc/src/api/client.rs`: `get_ask()`, `get_bid_object()`, `get_settlement()`, `get_bids_for_ask(ask_id, status_filter?)`, `get_open_asks(buyer?, page_size?)`, `get_vaults(owner)`. Each returns native `Object` types after proto conversion.
- **SDK marketplace query methods.** Added to `SomaClient` in `sdk/src/lib.rs`: `get_ask()`, `get_bid_object()`, `get_settlement()`, `get_bids_for_ask()`, `get_open_asks()`, `get_vaults()`. Each delegates to the inner `rpc::Client` via `Arc<RwLock<Client>>`.
- **Wired `soma accept --cheapest`.** Replaced the stub error message in `cli/src/commands/accept.rs` `execute_cheapest()` with live functionality: fetches all pending bids via `get_bids_for_ask(ask_id, Some("Pending"))`, sorts by price ascending, accepts up to `count` cheapest bids sequentially. Each AcceptBid auto-selects the richest USDC coin and gas coin fresh (since balances change after each accept). Prints progress per bid.
- **CLI seller listen.** Added `soma ask list [--buyer <addr>] [--limit N] [--json]` for sellers to discover open asks. Added `soma ask info <ask-id> [--json]` for detailed ask inspection. Added `soma ask listen [--interval 5s] [--json]` — polling loop that watches for new open asks and prints them as they appear. Ctrl-C to stop. `--json` outputs one JSON line per new ask for agent consumption.
- **CLI buyer listen.** Added `soma bid list --ask <ASK_ID> [--mine] [--status Pending] [--json]` for buyers to see all bids on their ask. Added `soma bid listen <ask-id> [--interval 5s] [--json]` — polling loop that watches for new bids on a specific ask and prints them as they appear. `--json` outputs one JSON line per new bid.
- **Design decisions:**
  - **Polling over streaming for v1.** The `SubscribeAsks` and `SubscribeBids` gRPC streaming RPCs exist in the proto and return empty streams. The CLI `listen` commands use polling (`get_open_asks` / `get_bids_for_ask` on an interval) as the reliable v1 approach. When the streaming infrastructure is wired in Phase 6 (broadcast channel from checkpoint processing), the CLI can switch to the streaming RPCs for lower latency. Polling is the correct v1 choice: simpler, more debuggable, works with any RPC infrastructure.
  - **Secondary indexes over full-table scan.** The `bids_by_ask` and `open_asks` RocksDB tables use the same checkpoint/tx indexing hooks as the owner and balance indexes. This gives O(1) lookup per ask_id → bid_ids, vs scanning all shared objects. Bounded by the live object set (entries removed when objects are deleted/status changes).
  - **Status filter via BCS deserialization.** `GetBidsForAsk` with `status_filter` deserializes each bid's BCS contents to check `bid.status`. This is O(n) in the number of bids per ask, which is acceptable given asks typically have <100 bids. A status-aware secondary index (e.g., `bids_by_ask_status`) would avoid this but adds complexity for marginal benefit.
- **Workspace compiles cleanly:** `cargo check` (normal) and `RUSTFLAGS="--cfg msim" cargo check` both pass. All types (234), authority (182), cli (25 lib + 15 integration), sdk (24) tests pass.

**What Phase 6 did so far:**
- **USDC genesis support.** Added `usdc_amounts: Vec<u64>` to `AccountConfig` (with `#[serde(default)]` and `Default` derive for backward compat). Added `UsdcAllocation` struct and `usdc_allocations` field to `TokenDistributionSchedule`. Genesis builder creates `CoinType::Usdc` coins from USDC allocations. Added `TestClusterBuilder::with_usdc_for_accounts(amount, count)` for easy test setup.
- **RPC ObjectType update.** Updated `rpc::types::ObjectType` to match domain types: `Coin(CoinType)` (discriminates SOMA/USDC), `Ask`, `Bid`, `Settlement`, `SellerVault`, `PendingWithdrawal`. Removed stale `Target` variant. Updated proto string conversion: `"Coin(SOMA)"`, `"Coin(USDC)"`, and all marketplace types. Updated all `types_conversions.rs` mappings (rpc types ↔ domain types). Fixed SDK and CLI `ObjectType::Coin` filter strings.
- **Proto transaction definitions.** Added marketplace proto messages (`CreateAsk`, `CancelAsk`, `CreateBid`, `AcceptBid`, `RateSeller`, `WithdrawFromVault`) and bridge proto messages (`BridgeDeposit`, `BridgeWithdraw`, `BridgeEmergencyPause`, `BridgeEmergencyUnpause`) to `transaction.proto` (field numbers 34-43). Added full round-trip conversions: domain → proto (in `rpc_proto_conversions.rs`), proto → rpc types (in `proto/soma/transaction.rs`), and rpc types → domain (in `types_conversions.rs`). Added corresponding `TransactionKind` variants and arg structs to `rpc::types::transaction`.
- **10 marketplace e2e tests** in `e2e-tests/tests/marketplace_tests.rs`:
  1. `test_marketplace_happy_path` — Full ask→bid→accept flow. Verifies Ask/Bid/Settlement/Vault objects, fee deduction, status transitions.
  2. `test_multi_bid_competition` — Ask with num_bids_wanted=3, 3 sellers bid at different prices, buyer accepts all 3. Verifies ask Filled with 3 accepted bids.
  3. `test_cancel_ask` — Create and cancel ask. Verifies status Cancelled.
  4. `test_negative_seller_rating` — Full flow + RateSeller. Verifies settlement rating changes to Negative.
  5. `test_seller_cannot_bid_own_ask` — Self-bid correctly rejected.
  6. `test_accept_wrong_coin_type` — AcceptBid with SOMA coin instead of USDC correctly rejected.
  7. `test_value_fee_accounting` — Verifies settlement amount and vault balance match exactly `bid.price - value_fee` (2.5% default).
  8. `test_vault_withdrawal` — Seller withdraws full vault balance, verifies USDC coin created with correct amount.
  9. `test_cancel_after_accept_rejected` — CancelAsk after accepting a bid correctly rejected (AskHasAcceptedBids).
  10. `test_rpc_marketplace_queries` — End-to-end ask→bid→accept flow with object state verification via `get_object` RPC at each step.
- **Fixed: marketplace RPC endpoint field masks.** The marketplace-specific RPC endpoints (`GetAsk`, `GetBidsForAsk`, `GetOpenAsks`, `GetSettlement`, `GetVault`) were returning objects with missing `previous_transaction` field. Fixed by adding `"previous_transaction"` to all read masks in `state_service/mod.rs` (`lookup_object_by_id()` helper and all inline masks). All marketplace RPC endpoints now return complete objects. Validated by `test_secondary_index_consistency` e2e test.
- **8 additional e2e tests** (tests 11-18) in `e2e-tests/tests/marketplace_tests.rs`:
  11. `test_transfer_soma_single` — Transfer SOMA single-coin single-recipient. Verifies created coin has correct CoinType::Soma and amount.
  12. `test_transfer_usdc_single` — Transfer USDC single-coin single-recipient with separate SOMA gas coin. Verifies created coin has CoinType::Usdc.
  13. `test_transfer_multi_recipient` — Transfer with per-recipient amounts (split). 2 recipients get different amounts from one coin.
  14. `test_merge_coins_soma` — MergeCoins with 2 SOMA coins. Verifies first coin has combined balance.
  15. `test_merge_coins_usdc` — MergeCoins with 2 USDC coins with separate SOMA gas. Verifies combined balance and CoinType::Usdc.
  16. `test_transfer_mixed_coin_types_rejected` — Transfer with both SOMA and USDC coins in the input set correctly rejected (InvalidArguments: coin type mismatch).
  17. `test_merge_mixed_coin_types_rejected` — MergeCoins with SOMA + USDC coins correctly rejected.
  18. `test_ask_expiry` — Creates ask with 10s timeout (minimum), advances 3 epochs (5s each) so epoch_start_timestamp_ms exceeds timeout. Bid on expired ask correctly rejected with AskExpired.
- **Fixed: USDC supply conservation bug.** The supply conservation check in `authority/src/authority.rs` `check_soma_conservation()` was summing ALL coin balances (including `CoinType::Usdc`) but comparing against `TOTAL_SUPPLY_SOMA`. USDC coins minted at genesis or via bridge deposits caused a `SUPPLY CONSERVATION VIOLATION` panic at epoch boundaries. Fixed by changing `ObjectType::Coin(_)` match to `ObjectType::Coin(CoinType::Soma)`, with an explicit `ObjectType::Coin(CoinType::Usdc) => {}` no-op arm. This unblocks all tests that combine USDC with epoch transitions.
- **Test 19: `test_usdc_epoch_supply_conservation`.** Explicitly validates the fix: creates a test cluster with both `with_usdc_for_accounts` AND `with_epoch_duration_ms(5_000)`, runs a full marketplace flow (ask→bid→accept with USDC payment), then waits for 2 epoch transitions. The supply conservation check runs at each epoch boundary and passes without panic.
- **Test 20: `test_default_positive_rating`.** Uses custom `MarketplaceParameters` with `rating_window_ms: 10_000` (10s) and short epochs (5s). Creates ask→bid→accept, advances 3 epochs past the rating deadline, then verifies `RateSeller` is rejected with `RatingDeadlinePassed` and settlement remains `SellerRating::Positive`. Required adding `with_marketplace_params()` to `TestClusterBuilder` (threads through `GenesisConfig` → `GenesisBuilder` → `SystemState::create()`).
- **Test 21: `test_vault_accumulation`.** Single seller fulfills 2 separate asks at different prices (0.50 USDC, 0.30 USDC). Verifies each AcceptBid creates a separate SellerVault with correct balance (price - 2.5% fee). Withdraws from both vaults, verifies total USDC received matches sum of vault balances, and verifies vaults are deleted (zero balance → deleted).
- **Test 22: `test_secondary_index_consistency`.** Creates 2 asks, verifies both appear in `open_asks` index via `get_open_asks()` RPC. Seller bids on ask[0], verifies `get_bids_for_ask()` returns the bid. Buyer accepts → verifies ask[0] removed from `open_asks` but ask[1] remains. Verifies status filter (`Pending` vs `Accepted`) works correctly. Cancels ask[1] → verifies it disappears from `open_asks`.
- **Fixed: RPC field mask bug.** All marketplace RPC endpoints (`GetAsk`, `GetBid`, `GetSettlement`, `GetBidsForAsk`, `GetOpenAsks`, `GetVault`) were missing `previous_transaction` in their read masks, causing proto conversion errors when clients tried to deserialize the returned objects. Fixed by adding `"previous_transaction"` to all `FieldMask::from_paths()` calls in `state_service/mod.rs`. The `lookup_object_by_id()` helper and all inline read masks now include it.
- **Infrastructure: `TestClusterBuilder::with_marketplace_params()`.** New builder method that sets custom `MarketplaceParameters` in genesis. Threaded through: `GenesisConfig.marketplace_params: Option<MarketplaceParameters>` → `NetworkConfig::generate_with_rng` → `GenesisBuilder.with_marketplace_params()` → `SystemState::create()`. Defaults to `MarketplaceParameters::default()` when not set.
- **Bridge committee genesis infrastructure.** Added `bridge_committee: Option<BridgeCommittee>` to `GenesisConfig` (with `#[serde(default)]`). Added `with_bridge_committee()` to `GenesisBuilder`. Threaded through `NetworkConfig::generate_with_rng()` → `GenesisBuilder.build()` → `SystemState::create()`. Previously always used `BridgeCommittee::empty()`; now uses provided committee when available.
- **`TestClusterBuilder::with_bridge_committee(n)`.** Generates `n` committee members with equal voting power (10000/n each), random `SomaAddress` keys, dummy 33-byte ECDSA pubkeys (real ECDSA deferred to Phase 3b). Threshold defaults: deposit=3334, withdraw=3334, pause=450, unpause=6667. With 4 members (2500 each): 2 members exceed deposit/withdraw threshold, 1 exceeds pause, 3 exceed unpause.
- **6 bridge e2e tests** (tests 23-28) in `e2e-tests/tests/marketplace_tests.rs`:
  23. `test_bridge_deposit_mints_usdc` — BridgeDeposit with 2/4 committee members (5000 > 3334 threshold). Verifies USDC coin minted with correct amount and CoinType::Usdc.
  24. `test_bridge_deposit_nonce_replay_rejected` — First deposit succeeds, second deposit with same nonce rejected. Uses `execute_transaction_may_fail` for failure handling.
  25. `test_bridge_withdraw_e2e` — User withdraws 3 USDC from 10 USDC genesis coin. Verifies PendingWithdrawal created with correct sender/amount/eth_address/nonce, USDC coin balance reduced.
  26. `test_bridge_emergency_pause_unpause` — Full flow: pause (1 member, 2500 > 450), deposit rejected while paused, withdraw rejected while paused, unpause (3 members, 7500 > 6667), deposit succeeds after unpause.
  27. `test_bridge_deposit_insufficient_stake_rejected` — Deposit with only 1/4 members (2500 < 3334) correctly rejected.
  28. `test_bridge_deposit_withdraw_roundtrip` — Deposit 10 USDC via bridge, withdraw 4, verify remaining balance is 6, deposit again with new nonce succeeds.
- **E2e failure test pattern.** Bridge tests that expect execution failure use `wallet.sign_transaction()` + `wallet.execute_transaction_may_fail()` instead of `sign_and_execute_transaction()` (which panics on failure). Failure matching handles both execution-level (`effects.status().is_err()`) and orchestrator-level (`Err(e)`) rejection paths.
- **Workspace compiles cleanly:** `cargo check` (normal) and `RUSTFLAGS="--cfg msim" cargo check` both pass. All types (234), authority (182), e2e marketplace+bridge (28) tests pass.

**What Phase 5+6 continued (settlements index, reputation, accept --cheapest):**
- **Settlements secondary indexes.** Added `settlements_by_buyer: DBMap<(SomaAddress, ObjectID), ()>` and `settlements_by_seller: DBMap<(SomaAddress, ObjectID), ()>` to `IndexStoreTables` in `authority/src/rpc_index.rs`. Populated during `update_marketplace_index_entries` (on Settlement creation) and cleaned up in `remove_marketplace_index_entries` (on Settlement deletion). Added `settlements_by_buyer_iter()` and `settlements_by_seller_iter()` query methods using composite key range scans. Bumped `CURRENT_DB_VERSION` to 6.
- **RPC: GetSettlements endpoint.** Added `GetSettlements` RPC to `state_service.proto` with `buyer`, `seller`, and `page_size` fields. At least one of buyer/seller is required. When both are provided, uses buyer index and cross-checks seller via BCS deserialization. Handler in `state_service/mod.rs` follows the same pattern as `GetBidsForAsk`.
- **RpcIndexes trait extended.** Added `settlements_by_buyer(&SomaAddress) -> Vec<ObjectID>` and `settlements_by_seller(&SomaAddress) -> Vec<ObjectID>` to the `RpcIndexes` trait in `types/src/storage/read_store.rs`. Implemented on `RestReadStore` in `authority/src/storage.rs`.
- **RPC client + SDK methods.** Added `get_settlements(buyer?, seller?, page_size?)` to `rpc::Client` and `SomaClient`.
- **CLI: `soma settlements`.** New command with `--as buyer|seller` role filter, `--address`, `--limit`, `--json`. Default: shows settlements for active address as both buyer and seller (deduped). Table output shows settlement ID, buyer, seller, amount (USDC), rating.
- **CLI: `soma reputation [address]`.** New command computing reputation from settlement data. Fetches settlements as buyer and as seller, computes: buyer settlements count, volume spent, unique sellers (counterparty diversity), seller settlements count, volume earned, approval rate (% non-negative), negative rating count, unique buyers. `--json` outputs structured `ReputationSummary`.
- **E2E test 29: `test_accept_cheapest`.** Creates ask with `num_bids_wanted=3`, 3 sellers bid at prices [1.50, 0.80, 1.20]. Uses `get_bids_for_ask` RPC to find pending bids, sorts by price ascending, accepts the 2 cheapest (0.80 and 1.20). Verifies: cheapest 2 bids Accepted, most expensive (1.50) still Pending, `get_bids_for_ask(Pending)` filter returns only 1 remaining bid.
- **E2E test 30: `test_settlements_index`.** Creates 2 asks from buyer, accepted by 2 different sellers. Verifies `get_settlements(buyer=...)` returns 2, `get_settlements(seller=seller_a)` returns 1 with correct buyer/seller fields, `get_settlements(seller=seller_b)` returns 1.
- **Workspace compiles cleanly:** `cargo check` (normal) and `RUSTFLAGS="--cfg msim" cargo check` both pass. All types (234), authority (182), cli (25 lib), sdk (24), e2e marketplace+bridge (30) tests pass.

**What Phase 5+6 continued (GetProtocolFund, GetReputation RPC, CLI wiring):**
- **RPC: `GetProtocolFund` endpoint.** Added `GetProtocolFund` RPC to `state_service.proto` (no request params, returns `uint64 balance`). Handler reads `system_state.protocol_fund_balance()` directly — no indexes needed. Added `get_protocol_fund()` to `rpc::Client` (returns `u64`) and `SomaClient`.
- **RPC: `GetReputation` endpoint.** Added `GetReputation` RPC to `state_service.proto` with `address` param, returns `GetReputationResponse` with buyer/seller metrics (settlements, volume, unique counterparties, negative ratings, approval rate). Handler computes reputation server-side using `settlements_by_buyer` and `settlements_by_seller` indexes — iterates settlement objects, aggregates volume, counts unique counterparties, computes approval rate. Same logic the CLI previously did client-side, now server-side for efficiency.
- **SDK: `get_reputation()` and `get_protocol_fund()`.** Both added to `rpc::Client` and `SomaClient`. `get_reputation()` returns the raw proto `GetReputationResponse` for maximum flexibility. `get_protocol_fund()` returns `u64`.
- **CLI: `soma reputation` now uses server-side RPC.** Rewrote `cli/src/commands/reputation.rs` to call `client.get_reputation(&addr)` instead of fetching all settlements and computing client-side. Removes the `types::bid::Bid`, `types::object::ObjectType`, `types::settlement::Settlement` imports — the CLI no longer needs to deserialize settlement objects. Same output format (text and JSON).
- **E2E test 31: `test_get_protocol_fund`.** Creates ask→bid→accept flow, verifies `get_protocol_fund()` returns 0 before any marketplace activity, then returns exactly `bid_price * 250 / 10_000` (2.5% value fee) after one AcceptBid.
- **E2E test 32: `test_get_reputation_rpc`.** Buyer creates 2 asks accepted by 2 different sellers. Verifies: buyer has 2 settlements, 2 unique sellers, correct total volume; seller_a has 1 settlement, 1 unique buyer, 100% approval rate, correct volume; seller_b same. Also verifies seller_settlements = 0 for buyer and buyer_settlements = 0 for sellers (cross-role separation).
- **Workspace compiles cleanly:** `cargo check` (normal) and `RUSTFLAGS="--cfg msim" cargo check` both pass. All types (234), authority (182), cli (25 lib), sdk (24), e2e marketplace+bridge (32) tests pass.

**What session 4 did (2026-03-29) — ECDSA bridge verification:**
- **Real ECDSA ecrecover in bridge executor.** Upgraded `verify_committee_stake()` → `verify_committee_signatures()` in `authority/src/execution/bridge.rs`. For each bit set in `signer_bitmap`, extracts a 65-byte recoverable signature from `aggregated_signature`, ecrecovers the public key using `Keccak256`, and verifies it matches the committee member's registered `ecdsa_pubkey`. Uses fastcrypto's `Secp256k1RecoverableSignature::recover_with_hash::<Keccak256>()`. Wrong signatures are rejected with a `SomaError` explaining the mismatch.
- **Bridge message encoding.** Added to `types/src/bridge.rs`: `encode_bridge_message()` (PREFIX + type + version + nonce + chainID + payload), `encode_deposit_payload()`, `encode_withdraw_payload()`, `encode_emergency_payload()`. Message format matches what the Solidity contract will use for `ecrecover`. Constants: `BRIDGE_MESSAGE_PREFIX = b"SOMA_BRIDGE_MESSAGE"`, `BRIDGE_MESSAGE_VERSION = 1`, `SOMA_BRIDGE_CHAIN_ID = 1`. `BridgeMessageType` enum: `UsdcDeposit(0)`, `UsdcWithdraw(1)`, `EmergencyOp(2)`, `CommitteeUpdate(3)`. `EmergencyOpCode` enum: `Freeze(0)`, `Unfreeze(1)`.
- **Bridge signing utilities.** Added to `types/src/bridge.rs`: `sign_bridge_message()` (signs with Keccak256), `build_bridge_signatures()` (produces aggregated_signature + signer_bitmap from a set of signers), `generate_test_bridge_committee()` (creates committee with real Secp256k1KeyPairs). These utilities are used by both unit tests and e2e tests, and will be used by the bridge-node crate.
- **TestClusterBuilder uses real keypairs.** `with_bridge_committee(n)` now generates real secp256k1 keypairs via `generate_test_bridge_committee()`. Keypairs stored on `TestCluster.bridge_keypairs` (in BTreeMap iteration order) so tests can sign bridge messages. Replaced dummy `vec![0x02; 33]` pubkeys with real compressed public keys.
- **TestAuthorityBuilder.with_genesis_config().** New builder method that passes a custom `GenesisConfig` through `ConfigBuilder::with_genesis_config()`. Used by unit tests to create an authority with a real bridge committee in genesis.
- **3 new bridge unit tests with real ECDSA** (total: 11 bridge unit tests, 185 authority tests):
  - `test_bridge_deposit_with_real_ecdsa_signatures` — 4-member committee via genesis, 2 members sign deposit, USDC minted successfully.
  - `test_bridge_deposit_wrong_signature_rejected` — Signatures from wrong keypairs (member 2 signs for member 0's slot) are rejected.
  - `test_bridge_nonce_replay_with_real_ecdsa` — First deposit succeeds, second with same nonce rejected with `BridgeNonceAlreadyProcessed`.
- **All 6 bridge e2e tests updated for real ECDSA.** Added helper functions `sign_deposit()`, `sign_pause()`, `sign_unpause()` at the top of the test file. Each bridge e2e test now produces real signatures using `test_cluster.bridge_keypairs`. All 32 marketplace+bridge e2e tests pass.
- **Workspace compiles cleanly:** `cargo check` (normal) and `RUSTFLAGS="--cfg msim" cargo check` both pass. All types (234), authority (185), cli (24), sdk (24), protocol-config (5), e2e marketplace+bridge (32) tests pass.

**What session 3 fixed (2026-03-29):**
- **CLI snapshot tests updated.** Removed 3 dead snapshot tests (`test_model_help`, `test_model_commit_help`, `test_target_help`) referencing deleted `model`/`target` commands. Removed `test_model_commit_missing_args` (tested deleted `model commit` command). Accepted 15 new snapshots reflecting the marketplace CLI commands. All CLI tests now pass: 25 lib + 2 doc + 15 integration + 24 snapshot = 66 tests, 0 failures.
- **protocol-config snapshot tests fixed.** Added `macros` feature to `serde_with` dependency (needed for `skip_serializing_none` attribute macro — the crate compiled without tests because the macro was only used in test compilation paths). Accepted 15 new version snapshots (5 versions × 3 snapshot types). 5 tests pass.
- **e2e test compilation fixes.** Fixed `full_node_tests.rs` — removed dead `test_local_scoring_server_auto_start` test referencing deleted `with_local_scoring()` method. Fixed `supply_conservation_tests.rs` — added `usdc_amounts: vec![]` to `AccountConfig` constructions, removed `model_registry()` method calls (model system deleted), replaced `emission_per_epoch` with `current_distribution_amount`. Fixed `reconfiguration_tests.rs` — added `usdc_amounts: vec![]` to 4 `AccountConfig` constructions. Fixed `dynamic_committee_tests.rs` — same `AccountConfig` fix.
- **Full e2e suite passes.** All 66 e2e tests pass (0 failures, 1 ignored). Includes marketplace (32), checkpoint (3), determinism (5), dynamic committee (1), failpoint (13), full node (7), indexer (3), lock (1), multisig (1), protocol version tests, reconfiguration, rpc, shared object, simulator, supply conservation, transaction orchestrator.
- **All crate tests pass.** types (234), authority (185), cli (66), sdk (24+1), protocol-config (5), rpc (36), indexer-alt (29 ignored — require Postgres), graphql (21 ignored — require Postgres), indexer-kvstore (10+4 ignored). Zero failures across all runnable tests.

**What session 5 did (2026-03-29) — bridge-node crate scaffolding:**
- **Created `bridge-node` crate.** New workspace member with 9 modules, 15 unit tests. Added to `Cargo.toml` workspace members and dependencies. Compiles cleanly under both normal and msim builds.
- **Module: `error.rs`** — `BridgeError` enum with 12 variants: `ProviderError`, `TransientProviderError`, `TxNotFound`, `TxNotFinalized`, `DepositEventNotFound`, `InvalidSignature`, `InsufficientStake`, `DuplicateSignature`, `BridgePaused`, `NonceAlreadyProcessed`, `GrpcError`, `PeerConnectionFailed`, `ConfigError`, `Internal`, `Other(anyhow)`. `BridgeResult<T>` alias.
- **Module: `types.rs`** — `BridgeAction` enum (Deposit, Withdrawal, EmergencyPause, EmergencyUnpause, CommitteeUpdate) with `to_message_bytes()` producing canonical bridge message encoding via `types::bridge::encode_bridge_message()`. `DepositEvent` (parsed from Ethereum logs) and `ObservedWithdrawal` (from Soma checkpoints) with `to_bridge_action()` conversion. 5 tests: encoding determinism, all action types, signability with real ECDSA keys.
- **Module: `config.rs`** — `BridgeNodeConfig` with all operational parameters: bridge key path, Ethereum RPC URLs (multi-endpoint with fallback), contract address, Soma RPC URL, gRPC listen address, peer addresses, chain ID, poll interval, max log query range, auto-pause threshold, retry settings. Validation method.
- **Module: `eth_client.rs`** — `EthClient` using raw JSON-RPC via reqwest (not alloy — alloy 0.15 has serde `__private` incompatibility with serde 1.0.228 on Rust 1.93). Methods: `get_chain_id()`, `get_last_finalized_block_id()`, `get_deposit_events_in_range()`, `parse_deposit_log()`. Multi-endpoint rotation with failure tracking and threshold-based auto-pause detection. 4 tests: deposit log parsing, wrong contract rejection, short data handling, endpoint rotation.
- **Module: `eth_syncer.rs`** — `EthSyncer` with two concurrent tasks: finalized block poller (configurable interval, `watch` channel) and event listener (waits on `watch::Receiver::changed()`, queries in chunks with `-32005` / "too many results" range-halving retry). Returns `EthSyncerHandle` with task handles + event/finalized-block channels. Directly adapted from Sui's `eth_syncer.rs`.
- **Module: `server.rs`** — `BridgeServer` implementing gRPC `BridgeNode` trait. Signature collection in `DashMap<Vec<u8>, BTreeMap<u32, Vec<u8>>>` (action_digest → signer signatures). Methods: `has_quorum()` (checks collected stake vs threshold), `get_aggregated_signatures()` (builds concatenated sigs + bitmap for on-chain submission), `update_committee()` (clears stale sigs). 5 tests: submit/get, quorum detection, aggregated signature format, invalid signer, duplicate handling.
- **Module: `checkpoint_watcher.rs`** — `CheckpointWatcher` that scans checkpoint output objects for `PendingWithdrawal` (via `Object::deserialize_contents()`) and detects epoch boundaries. Produces `CheckpointEvent::NewWithdrawal` and `CheckpointEvent::EpochBoundary`. Designed for integration with `data-ingestion` framework. 1 test: epoch boundary detection.
- **Module: `node.rs`** — `BridgeNode` orchestrator. `run()` spawns: EthSyncer, deposit handler (signs + stores locally), withdrawal handler (signs + stores locally), checkpoint watcher. gRPC peer exchange and quorum-based on-chain submission are TODO (documented in code). Uses `Arc<Secp256k1KeyPair>` since keypairs don't implement Clone.
- **Module: `proto_generated.rs`** — Hand-written prost message types and tonic trait matching `proto/bridge.proto`. Avoids protoc build dependency (protoc not available in workspace). Full gRPC server/client codegen deferred to when protoc is available. Proto definition kept in `proto/bridge.proto` for reference.
- **Design decisions:**
  - **reqwest over alloy for Ethereum RPC.** alloy 0.15 has `serde::__private` incompatibility with serde 1.0.228 on Rust 1.93.0 (the workspace's pinned toolchain). Raw JSON-RPC via reqwest (already a workspace dep) is simpler and sufficient — we only need `eth_getBlockByNumber` and `eth_getLogs`. This can be upgraded to alloy when the serde issue is resolved in a future alloy release.
  - **Hand-written proto types.** protoc is not available in the build environment. The proto is small (5 messages), so hand-written prost types with the tonic `BridgeNode` trait are equivalent. The `proto/bridge.proto` file is kept as the source of truth; regenerate `proto_generated.rs` when protoc is available.
  - **Arc<Secp256k1KeyPair>** in the orchestrator. fastcrypto's `Secp256k1KeyPair` doesn't implement `Clone`, and the keypair is shared across deposit and withdrawal handler tasks. `Arc` wrapping is the correct pattern.
- **Workspace compiles cleanly:** `cargo check` (normal) and `RUSTFLAGS="--cfg msim" cargo check` both pass. 15 bridge-node tests pass. All existing crate tests unaffected.

**What session 6 did (2026-03-29) — bridge-node testing & retry infrastructure:**
- **Module: `retry.rs`** — `retry_with_backoff()` async utility implementing exponential backoff. Initial interval 400ms, 2x multiplier, max interval 120s, configurable max elapsed time. Only retries on `TransientProviderError` and `ProviderError`; all other errors return immediately. 4 unit tests: succeeds after transient failures, non-retryable error returns immediately, timeout exceeded, succeeds on first try.
- **Wired retry into EthClient.** Added `get_last_finalized_block_id_with_retry()` and `get_deposit_events_in_range_with_retry()` convenience methods. Wired `retry_with_backoff` into `EthSyncer`'s finalized block poller and event query paths. On retry exhaustion in poller, calls `rotate_endpoint()` to try next RPC URL.
- **EthClient wiremock tests (3 tests).** Added `wiremock` as dev-dependency. `test_get_last_finalized_block` — mock JSON-RPC `eth_getBlockByNumber("finalized")`, verify block number parsed. `test_get_deposit_events` — mock `eth_getLogs` with ABI-encoded deposit data, verify all fields parsed. `test_transient_error_detection` — mock `-32005` error response, verify `TransientProviderError` returned.
- **EthSyncer wiremock tests (4 tests).** `test_syncer_detects_finalized_block` — full syncer lifecycle: poller detects block 100, watch channel updated, test receives via `finalized_block_rx`. `test_syncer_emits_deposit_events` — syncer detects deposits and emits via `event_rx` channel with correct nonce/amount. `test_query_events_with_retry_range_halving` — range 0..2000 with max_range 500 correctly chunks into sub-queries. `test_query_events_with_retry_transient_error_halves` — first chunk returns `-32005`, range halves, subsequent chunks succeed.
- **Crypto cross-verification tests (8 tests in `types.rs`).** `test_sign_and_ecrecover_roundtrip` — sign deposit message, ecrecover pubkey, verify match. `test_ecrecover_wrong_message_fails` — sign message A, ecrecover with message B produces different key. `test_build_bridge_signatures_format` — verify aggregated signature layout (130 bytes for 2 sigs, bitmap 0b00000101), ecrecover each signature slice individually. `test_message_encoding_known_values` — fixed-input deposit encoding with exact byte-level verification (prefix, type, version, nonce, chainID, payload positions and lengths). `test_withdrawal_encoding_known_values` — same for withdrawal messages. `test_emergency_encoding_known_values` — freeze=0, unfreeze=1 payload bytes. `test_committee_update_encoding` — count + (pubkey + power) encoding. `test_all_action_types_ecrecover_roundtrip` — sign→ecrecover for all 5 action types (Deposit, Withdrawal, EmergencyPause, EmergencyUnpause, CommitteeUpdate).
- **Test vector note.** The `test_message_encoding_known_values` test pins the exact keccak256 hash of a deterministic deposit message (nonce=1, zero recipient, amount=1M). This hash MUST match what `keccak256(abi.encodePacked(...))` produces in Solidity with the same inputs. When the Solidity contract is written, add a Foundry regression test with this exact input and verify the hash matches.
- **Workspace compiles cleanly:** `cargo check` (normal) and `RUSTFLAGS="--cfg msim" cargo check` both pass. 34 bridge-node tests pass (was 15). All types (234), authority (185), cli (25), sdk (24) tests pass.

**Next task for a new implementer:**

1. **Phase 3b — Ethereum contract.** The bridge-node crate exists with 34 tests. Next:
   - **Ethereum contract** (Foundry project in `bridge/evm/`). 3 Solidity files adapting Sui's patterns. Use `forge init`, OpenZeppelin UUPS + Pausable. **Critical:** message encoding in Solidity MUST match `types::bridge::encode_bridge_message()` (PREFIX + type + nonce + chainID + payload, keccak256). The `test_message_encoding_known_values` test in `bridge-node/src/types.rs` provides exact byte-level verification — the Solidity encoder must produce the same keccak256 hash for the same inputs. Rust-side signing utilities (`sign_bridge_message()`, `build_bridge_signatures()`) are in `types/src/bridge.rs`.

2. **Phase 3b — remaining bridge-node work:**
   - **Mock EthClient for unit tests** — port Sui's `EthMockService` pattern for more granular control (current tests use wiremock HTTP mocks which are sufficient for integration but heavier).
   - **Full deposit round-trip test** — mock Ethereum deposit event → EthSyncer emits → bridge node signs → gRPC exchange → `BridgeDeposit` system tx submitted.
   - **Full withdrawal round-trip test** — `BridgeWithdraw` tx → checkpoint watcher detects `PendingWithdrawal` → signs → mock Ethereum contract call.

3. **Phase 6 remaining e2e tests** (requires Postgres):
   - **E2E: indexer pipeline** — verify soma_asks, soma_bids, soma_settlements, soma_vaults tables populated correctly after marketplace transactions.
   - **E2E: reputation queries** — after mixed positive/negative ratings, verify GraphQL reputation endpoint returns correct metrics.

4. **Phase 5 remaining work** (client layer — sorted by impact):
   - **RPC: `subscribe_asks`/`subscribe_bids` real streaming** — wire broadcast channels from checkpoint/tx processing into the gRPC streaming RPCs. Currently return empty streams; CLI uses polling as fallback.

**CLI command summary (current state):**
```
# Marketplace — Buyer commands
soma ask create --task <file> --max-price <usdc> [--timeout 5m] [--num-bids 1]
soma ask cancel <ask-id>
soma ask list [--buyer <addr>] [--limit 100] [--json]
soma ask info <ask-id> [--json]
soma accept <bid-id>                              # auto-selects USDC coin
soma accept --ask <ask-id> --cheapest [--count N] # auto-selects N cheapest pending bids
soma rate <settlement-id>                         # negative seller rating
soma bid listen <ask-id> [--interval 5s] [--json] # poll for new bids on your ask

# Marketplace — Seller commands
soma ask listen [--interval 5s] [--json]          # poll for new open asks (discover work)
soma bid create <ask-id> --price <usdc> --response <file>
soma bid list --ask <ask-id> [--mine] [--status Pending] [--json]
soma vault withdraw <vault-id> [--amount <usdc>]
soma settlements [--as buyer|seller] [--address 0xABC] [--json]
soma reputation [<address>] [--json]

# Transfers
soma transfer <amount> <recipient>                # SOMA (default)
soma transfer <amount> <recipient> --usdc         # USDC
soma transfer 0 <r1> <r2> --amounts 3,2           # multi-recipient
soma merge                                        # consolidate dust coins

# Staking & gas
soma stake --validator <addr> --amount <soma>
soma unstake <staked-soma-id>
soma faucet
soma balance [<address>]
soma status

# Management
soma wallet {list, new, switch, ...}
soma env {list, switch, new}
soma validator {display-metadata, list, ...}
soma objects {list, get}
soma tx <digest>
```

**Sui bridge reference files (MystenLabs/sui repo):**
- `crates/sui-bridge/src/eth_client.rs` — EthClient with alloy, `get_last_finalized_block_id()`, `get_events_in_range()`
- `crates/sui-bridge/src/eth_syncer.rs` — EthSyncer with finalized block poller + event listener, `-32005` retry
- `crates/sui-bridge/src/crypto.rs` — Secp256k1 signing/verification with Keccak256, `to_eth_address()`
- `crates/sui-bridge/src/server/handler.rs` — BridgeRequestHandler, verify then sign pattern
- `crates/sui-bridge/src/error.rs` — BridgeError enum
- `crates/sui-bridge/src/types.rs` — BridgeAction, SignedBridgeAction
- `bridge/evm/contracts/SuiBridge.sol` — Main bridge contract (deposit, withdraw, emergency)
- `bridge/evm/contracts/BridgeCommittee.sol` — ECDSA signature verification
- `bridge/evm/contracts/utils/BridgeUtils.sol` — Message encoding, hash computation, stake thresholds
- `bridge/evm/contracts/utils/MessageVerifier.sol` — Nonce tracking modifier
- `bridge/evm/test/SuiBridgeTest.t.sol` — Foundry tests with regression test patterns

### Phase 1: Strip

Remove everything that's going away. Goal: compiles with model/target/submission/python code gone.

- [x] **Delete dead crates from workspace.** Remove `scoring` (model evaluation engine), `models` (V1 transformer architecture), `arrgen` (array generation for model architectures) from `Cargo.toml` members and delete their directories. Verify no remaining crate depends on `arrgen` before deleting.
- [x] **Delete Python SDK and Python models.** Remove `python-sdk/` (PyO3 wrapper), `python-examples/` (SDK usage examples), `python-models/` (Python model code) directories entirely. Remove `python-sdk` from workspace `Cargo.toml` members. The CLI replaces all Python SDK functionality.
- [x] **Delete dead type files.** Remove `types/src/target.rs`, `types/src/model.rs`, `types/src/submission.rs`, `types/src/tensor.rs` (SomaTensor, only used for embeddings/distances). Remove their `mod` declarations from `types/src/lib.rs`. Note: `metadata.rs` was kept — it's used by the `blobs` crate for generic blob downloading (checksums, sizes), not model-specific.
- [x] **Delete dead system state modules.** Remove `types/src/system_state/model_registry.rs` (model registration and staking pools) and `types/src/system_state/target_state.rs` (target generation, difficulty adjustment, EMA). Remove `model_registry` and `target_state` fields from `SystemStateV1` struct and `create()`/`advance_epoch()` methods.
- [x] **Remove dead transaction variants.** Delete all model/submission/target variants from `TransactionKind` enum in `types/src/transaction.rs`: `CreateModel`, `CommitModel`, `RevealModel`, `AddStakeToModel`, `SetModelCommissionRate`, `DeactivateModel`, `ReportModel`, `UndoReportModel`, `SubmitData`, `ClaimRewards`, `ReportSubmission`, `UndoReportSubmission`. Delete their corresponding arg structs.
- [x] **Replace TransferCoin/PayCoins with unified Transfer.** Replace the two coin transfer variants with a single `Transfer { coins: Vec<ObjectRef>, amounts: Option<Vec<u64>>, recipients: Vec<SomaAddress> }` and add `MergeCoins { coins: Vec<ObjectRef> }` for explicit merge-only operations. Update all match arms, serialization, and the transaction builder.
- [x] **Delete dead execution files.** Remove `authority/src/execution/model.rs` and `authority/src/execution/submission.rs`. Remove their arms from `create_executor()` in `execution/mod.rs`.
- [x] **Simplify staking execution.** In `authority/src/execution/staking.rs`, remove `AddStakeToModel` handling — only validator staking remains (`AddStake`, `WithdrawStake`).
- [x] **Simplify epoch change.** In `authority/src/execution/change_epoch.rs`, remove target reward calculation, model reward distribution, difficulty adjustment, target generation. All SOMA emissions + accumulated gas fees go to validators by stake weight. No USDC distribution at epoch boundary.
- [x] **Clean protocol-config.** Remove all target params (distance threshold, initial targets, hits per epoch, EMA decay, reward allocation bps, submitter/model/claimer reward splits), all model params (architecture version, commit/reveal windows, minimum stake), all submission params (bond per byte, max data size), `SomaTensor` type. Change `TOTAL_SUPPLY_SOMA` from 10M to 1B. Also removed `DistanceExceedsThreshold` error variant, `tensor.rs`, and `burn` dependency from protocol-config.
- [x] **Clean digests and crypto.** In `digests.rs`, remove `ModelWeightsCommitment`, `DecryptionKeyCommitment`, `EmbeddingCommitment`, `DataCommitment`. In `crypto.rs`, remove `DecryptionKey` and related types. Keep Ed25519/BLS signing infrastructure.
- [x] **Fix all compilation errors and verify `cargo check` passes.** Chase down every broken import, match arm, and reference. This will touch many files across the workspace — RPC, GraphQL, CLI, SDK, tests, indexer.

### Phase 2: New types

- [x] **Add CoinType enum.** In `types/src/object.rs`, add `pub enum CoinType { Soma, Usdc }` with Serialize/Deserialize/Clone/Copy/PartialEq/Eq/Hash derives. Change `ObjectType::Coin` to `ObjectType::Coin(CoinType)`. Update every match arm on `ObjectType` across the codebase (authority, RPC, indexer, CLI, tests).
- [x] **Add TaskDigest and ResponseDigest to digests.rs.** Create `pub struct TaskDigest(Digest)` and `pub struct ResponseDigest(Digest)` following the exact pattern of `TransactionDigest` — newtype over `Digest`, Base58 serialization, `FromStr`, `Display`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash`, `Ord`, `Serialize`, `Deserialize`, `JsonSchema`. Add `AskId`, `BidId`, `SettlementId` as `pub type` aliases for `ObjectID`.
- [x] **Create Ask type.** New file `types/src/ask.rs` with `Ask` struct (id, buyer, task_digest, max_price_per_bid, num_bids_wanted, timeout_ms, created_at_ms, status, accepted_bid_count), `AskStatus` enum (Open, Filled, Cancelled, Expired). BCS serialization.
- [x] **Create Bid type.** New file `types/src/bid.rs` with `Bid` struct (id, ask_id, seller, price, response_digest, created_at_ms, status), `BidStatus` enum (Pending, Accepted, Rejected, Expired). BCS serialization.
- [x] **Create Settlement type.** New file `types/src/settlement.rs` with `Settlement` struct (id, ask_id, bid_id, buyer, seller, amount, task_digest, response_digest, settled_at_ms, seller_rating, rating_deadline_ms), `SellerRating` enum (Positive, Negative). No buyer rating — acceptance IS the signal.
- [x] **Create SellerVault type.** New file `types/src/vault.rs` with `SellerVault` struct (id, owner, balance). Owned object — only the seller writes.
- [x] **Add marketplace transaction variants.** In `TransactionKind`, add: `CreateAsk(CreateAskArgs)`, `CancelAsk { ask_id: AskId }`, `CreateBid(CreateBidArgs)`, `AcceptBid(AcceptBidArgs)`, `RateSeller { settlement_id: SettlementId }` (negative only — no bool param), `WithdrawFromVault { vault: ObjectRef, amount: Option<u64>, recipient_coin: Option<ObjectRef> }`. Create corresponding arg structs using `TaskDigest`/`ResponseDigest` (not raw `[u8; 32]`).
- [x] **Add MarketplaceParameters to SystemStateV1.** Add `marketplace_params: MarketplaceParameters` (rating_window_ms, min_ask_timeout_ms, max_ask_timeout_ms, marketplace_fee_bps) and `protocol_fund_balance: u64` to `SystemStateV1`. Wire into `create()` and `advance_epoch()`. MarketplaceParameters lives in `types/src/bridge.rs` alongside BridgeState.
- [x] **Add BridgeState to SystemStateV1.** Add `bridge_state: BridgeState` with `BridgeCommittee`, `BridgeMember` (ecdsa_pubkey as `Vec<u8>`, voting_power), `PendingWithdrawal` type, deposit/withdraw nonce tracking, pause state. Created `types/src/bridge.rs` for these types. Note: `BridgeMember.ecdsa_pubkey` uses `Vec<u8>` (not `[u8; 33]`) for serde compatibility.
- [x] **Add bridge transaction variants.** Add `BridgeDeposit(BridgeDepositArgs)`, `BridgeWithdraw(BridgeWithdrawArgs)`, `BridgeEmergencyPause(BridgeEmergencyPauseArgs)`, `BridgeEmergencyUnpause(BridgeEmergencyUnpauseArgs)` to `TransactionKind`. Bridge deposit/pause/unpause are gasless system transactions (like `ConsensusCommitPrologueV1`).
- [x] **Add Secp256k1 bridge key to ValidatorInfo.** Add `bridge_ecdsa_pubkey: Option<Vec<u8>>` and `next_epoch_bridge_ecdsa_pubkey: Option<Vec<u8>>` to `ValidatorMetadata` (with `#[serde(default)]` for backward compat). Note: uses `Vec<u8>` not `[u8; 33]` for serde.
- [x] **Update genesis config.** Replaced `emission_per_epoch` with `emission_initial_distribution_amount`, `emission_period_length`, `emission_decrease_rate` in `GenesisCeremonyParameters`. Defaults: 100K SOMA/epoch, 10 epoch periods, 10% decay. MarketplaceParameters and BridgeCommittee passed to SystemState::create().
- [x] **Rewrite EmissionPool.** Replace flat `emission_per_epoch` with geometric step-decay: `balance`, `distribution_counter`, `current_distribution_amount`, `period_length`, `decrease_rate`. Each epoch: emit `min(current_distribution_amount, balance)`. Every `period_length` epochs, reduce by `decrease_rate` bps. Default: 100K SOMA/epoch, 10 epoch periods, 10% decay.
- [x] **Add object type handling.** Register Ask, Bid, Settlement, SellerVault, PendingWithdrawal in `ObjectType` enum and object creation/deserialization paths. Updated Display/FromStr impls for all new variants.
- [x] **Verify `cargo check` passes.** Both normal and msim builds pass. 234 types crate unit tests pass.

### Phase 3: Execution

- [x] **Rewrite coin executor for unified Transfer.** (Done in Phase 2) Coin executor in `authority/src/execution/coin.rs` already validates all input coins share the same `CoinType`, extracts the type from input coins, and propagates it to new coin objects. `MergeCoins` also validates same `CoinType`. Staking executor rejects non-SOMA coins.
- [x] **Create marketplace executor.** New file `authority/src/execution/marketplace.rs`. Implements CreateAsk, CancelAsk, CreateBid (all validation per spec), AcceptBid (atomic settlement with fee deduction, vault creation, settlement creation), RateSeller (negative only), WithdrawFromVault (with vault deletion on zero balance). Ask, Bid, Settlement created as shared objects. SellerVault created per-settlement as owned object (see design note in progress summary).
- [x] **Implement AcceptBid as atomic settlement.** (Included in marketplace executor above.)
- [x] **Implement RateSeller (negative only).** (Included in marketplace executor above.)
- [x] **Implement WithdrawFromVault.** (Included in marketplace executor above.)
- [x] **Create bridge executor.** New file `authority/src/execution/bridge.rs`. Implements BridgeDeposit (stake threshold via signer_bitmap, mint USDC, nonce tracking), BridgeWithdraw (burn USDC, create PendingWithdrawal as immutable), BridgeEmergencyPause/Unpause (asymmetric thresholds). Full ECDSA ecrecover deferred to Phase 3b. Modeled after Sui's native bridge.
- [x] **Wire executors into dispatch.** Updated `create_executor()` in `execution/mod.rs` — MarketplaceExecutor and BridgeExecutor replace todo!() placeholders.
- [x] **Simplify epoch change.** (Done in Phase 1/2) All SOMA emissions (geometric decay) + accumulated SOMA gas fees → validators by stake weight. USDC value fees already in protocol_fund_balance (no epoch distribution).
- [x] **Unit tests: marketplace happy paths.** 21 tests in `authority/src/unit_tests/marketplace_tests.rs`. Full ask→bid→accept→rate→withdraw flow tested. Settlement creation, vault crediting, fee deduction, ask filling all verified.
- [x] **Unit tests: marketplace edge cases.** Self-bid, double-rate, over-accept, wrong CoinType, insufficient balance, wrong sender, insufficient vault balance all tested and rejected correctly.
- [x] **Unit tests: bridge.** 8 tests in `authority/src/unit_tests/bridge_tests.rs`. Deposit/withdraw/pause/unpause paths tested. Insufficient stake, nonce replay, wrong coin type, insufficient balance all tested.

### Phase 3b: Bridge infrastructure

**Architecture reference:** Sui's `sui-bridge` crate (`crates/sui-bridge/`) and `bridge/evm/` Solidity contracts. Our bridge is substantially simpler: USDC-only (no multi-token config, no decimal conversion, no rate limiting). Key Sui patterns to adopt directly:

- **EthClient** (`eth_client.rs`): wraps `alloy::providers::Provider`. `get_last_finalized_block_id()` calls `eth_getBlockByNumber("finalized", false)`. `get_events_in_range(address, start, end)` uses `Filter` with `from_block`/`to_block`. Callsite responsible for chunking.
- **EthSyncer** (`eth_syncer.rs`): two concurrent tasks — finalized block poller (5s interval, `watch` channel) and per-contract event listener. Event listener waits on `watch::Receiver::changed()`, queries in chunks of `ETH_LOG_QUERY_MAX_BLOCK_RANGE` (1000 blocks), halves range on `-32005` errors.
- **BridgeCommittee.sol**: ECDSA `ecrecover` + bitmap to sum stake. `verifySignatures()` recovers signer from each signature, checks not blocklisted, sums stake, rejects if < required.
- **MessageVerifier.sol**: `verifyMessageAndSignatures` modifier — checks message type, verifies sigs via committee, increments nonce (except TOKEN_TRANSFER which uses per-transfer nonces).
- **SuiBridge.sol**: `bridgeERC20()` emits `TokensDeposited` event with auto-incrementing nonce. `transferBridgedTokensWithSignatures()` verifies ECDSA sigs + nonce not replayed, transfers from vault. `executeEmergencyOpWithSignatures()` for pause/unpause.
- **Crypto** (`crypto.rs`): `BridgeAuthorityKeyPair = Secp256k1KeyPair` from fastcrypto. Signs with `sign_recoverable_with_hash::<Keccak256>`. Verifies with `verify_recoverable_with_hash::<Keccak256>`. Our executor already has `verify_committee_stake()` with bitmap — Phase 3b adds real ECDSA ecrecover using fastcrypto's existing `Secp256k1PublicKey`.
- **Handler** (`server/handler.rs`): HTTP/axum server (not gRPC despite the name). `BridgeRequestHandler` has `verify_eth` (fetches from EthClient) and `verify_sui` (fetches from SuiClient), then signs with ECDSA key. We simplify to gRPC (tonic, already in workspace deps).
- **Message format**: Sui uses `MESSAGE_PREFIX + messageType + version + nonce + chainID + payload` (abi.encodePacked), hashed with keccak256 for ECDSA signing. We adopt the same pattern for Ethereum-side verification.

**What we can reuse directly from Sui:**
1. `EthClient` pattern — nearly 1:1 with alloy. Only simplification: one contract address (not a set), one event type (deposit).
2. `EthSyncer` pattern — finalized block poller + event listener architecture. Identical logic.
3. `BridgeCommittee.sol` — `verifySignatures()` is exactly what we need. Simplify by removing blocklist, BridgeConfig dependency, and multi-token logic.
4. `MessageVerifier.sol` — `verifyMessageAndSignatures` modifier with nonce tracking. Reuse directly.
5. `BridgeUtils.sol` — `encodeMessage()`, `computeHash()`, message struct. Simplify to 3 message types (DEPOSIT, WITHDRAW, EMERGENCY_OP) instead of 8.
6. `crypto.rs` — fastcrypto `Secp256k1KeyPair` signing/verification with Keccak256 hash. Already available in our workspace.
7. Test patterns — `EthMockService` mock provider, `mock_last_finalized_block()`, `mock_get_logs()`. Foundry tests with `getSignature()`, `vm.expectRevert()`.

**What we simplify vs Sui:**
- No multi-token support (BridgeConfig, BridgeVault, decimal conversion, token prices) — USDC only
- No rate limiting (BridgeLimiter, daily limits, USD value computation) — not in v1
- No blocklisting — not in v1
- No upgrade mechanism via bridge messages — standard proxy upgrade instead
- No SuiClient/SuiSyncer (Sui watches Move events) — we watch checkpoint objects directly via existing data-ingestion framework
- gRPC signature exchange instead of HTTP/axum (matches our existing tonic stack)

#### Ethereum contract (Solidity) — Foundry project in `bridge/evm/`

- [ ] **Project setup.** `forge init bridge/evm`. Add OpenZeppelin contracts-upgradeable. Configure foundry.toml with solc 0.8.20+, remappings for OZ.
- [ ] **SomaBridgeMessage.sol** — Library (replaces BridgeUtils). 3 message types: `USDC_DEPOSIT = 0`, `USDC_WITHDRAW = 1`, `EMERGENCY_OP = 2`, `COMMITTEE_UPDATE = 3`. `MESSAGE_PREFIX = "SOMA_BRIDGE_MESSAGE"`. `encodeMessage()` using abi.encodePacked (same as Sui). `computeHash()` via keccak256. Stake thresholds: DEPOSIT/WITHDRAW = 3334 (f+1), FREEZING = 450, UNFREEZING = 6667, COMMITTEE_UPDATE = 6667. Deposit payload: `bytes32 somaRecipient + uint64 amount`. Withdraw payload: `address ethRecipient + uint64 amount`. Emergency payload: `uint8 opCode` (0=freeze, 1=unfreeze).
- [ ] **SomaBridgeCommittee.sol** — Adapted from Sui's BridgeCommittee.sol. `verifySignatures(bytes[] sigs, Message msg)`: ecrecover each sig, sum stake, reject if < required. `initialize(address[] committee, uint16[] stake, uint16 minStake)`. No blocklist in v1. No BridgeConfig dependency (no multi-token). Committee update via `updateCommitteeWithSignatures()` — verifies old committee sigs, replaces members/stakes.
- [ ] **SomaBridge.sol** — Main contract (replaces SuiBridge.sol). `deposit(bytes32 somaRecipient, uint64 amount)`: transfers USDC from sender to contract via SafeERC20, emits `Deposited(nonce, sender, somaRecipient, amount)`, increments nonce. `withdrawWithSignatures(bytes[] sigs, Message msg)`: verifies committee sigs, checks nonce not replayed (`isWithdrawalProcessed` mapping), transfers USDC to recipient, marks processed. `executeEmergencyOpWithSignatures(bytes[] sigs, Message msg)`: pause/unpause. `updateCommitteeWithSignatures(bytes[] sigs, Message msg)`: rotates committee at epoch boundaries. Inherits PausableUpgradeable + UUPS proxy.
- [ ] **Contract tests (Foundry).** `SomaBridgeTest.t.sol`: deposit emits correct event with incrementing nonce, withdraw with valid sigs releases USDC, withdraw with insufficient stake reverts, nonce replay reverts, pause then all ops revert, unpause requires high threshold, committee update rotates members. `SomaBridgeCommitteeTest.t.sol`: signature verification, duplicate sig detection, stake threshold enforcement. Regression tests with known keypair/signature/message tuples (same pattern as Sui's `testTransferSuiToEthRegressionTest`).
- [ ] **Deploy script for Sepolia.** Forge script deploying UUPS proxies for Committee and Bridge. Configure with 4 test committee members, USDC token address on Sepolia.

#### Bridge node crate (`bridge-node/`) — new workspace member

- [x] **Crate setup.** Created `bridge-node/Cargo.toml` with deps: `reqwest` (raw JSON-RPC — alloy 0.15 has serde incompatibility with Rust 1.93), `tonic`, `prost` (gRPC), `fastcrypto` (Secp256k1), `types`, `tokio`, `tracing`, `serde`/`serde_json`, `backoff`, `dashmap`, `hex`. Added to workspace members and dependencies. 9 modules: `eth_client`, `eth_syncer`, `error`, `types`, `server`, `checkpoint_watcher`, `config`, `node`, `proto_generated`.
- [x] **Error types** (`error.rs`). `BridgeError` enum with 12 variants covering Ethereum RPC, signatures, bridge state, gRPC, config, and generic errors. `BridgeResult<T>` alias.
- [x] **Bridge action types** (`types.rs`). `BridgeAction` enum (Deposit, Withdrawal, EmergencyPause, EmergencyUnpause, CommitteeUpdate) with `to_message_bytes()` producing canonical encoding via `types::bridge::encode_bridge_message()`. `DepositEvent` and `ObservedWithdrawal` types. 5 unit tests.
- [x] **Crypto.** Signing utilities already exist in `types/src/bridge.rs` (`sign_bridge_message`, `build_bridge_signatures`, `generate_test_bridge_committee`). No separate `crypto.rs` module needed — the bridge-node crate uses those directly.
- [x] **EthClient** (`eth_client.rs`). Raw JSON-RPC via reqwest. `get_chain_id()`, `get_last_finalized_block_id()`, `get_deposit_events_in_range()`, `parse_deposit_log()` (ABI-decoded deposit events). Multi-endpoint rotation with failure tracking. 4 unit tests.
- [x] **EthSyncer** (`eth_syncer.rs`). Two concurrent tasks: finalized block poller (`watch` channel) and event listener (waits on `changed()`, chunks queries, halves range on `-32005`). Returns `EthSyncerHandle`.
- [x] **gRPC signature exchange** (`server.rs` + `proto/bridge.proto` + `proto_generated.rs`). `BridgeServer` with `DashMap` signature collection. `has_quorum()`, `get_aggregated_signatures()` (concatenated sigs + bitmap), `update_committee()`. Proto types hand-written (protoc not available). 5 unit tests.
- [x] **Checkpoint watcher** (`checkpoint_watcher.rs`). Scans checkpoint objects for `PendingWithdrawal`, detects epoch boundaries. `CheckpointEvent` enum. 1 unit test.
- [x] **Bridge node orchestrator** (`node.rs`). `BridgeNode::run()` spawns EthSyncer, deposit handler, withdrawal handler. Local signature collection works; peer gRPC exchange and quorum-based submission are TODO.
- [x] **ECDSA verification in bridge executor.** Upgraded `verify_committee_stake()` → `verify_committee_signatures()` in `authority/src/execution/bridge.rs`. For each bit set in signer_bitmap: extracts 65-byte recoverable signature from `aggregated_signature`, ecrecovers public key using `Secp256k1RecoverableSignature::recover_with_hash::<Keccak256>`, verifies it matches the committee member's `ecdsa_pubkey`. Message encoding: `SOMA_BRIDGE_MESSAGE || type(1) || version(1) || nonce(8,BE) || chainID(8,BE) || payload` — matching the planned Solidity encoding. Bridge signing utilities (`sign_bridge_message`, `build_bridge_signatures`, `generate_test_bridge_committee`) added to `types/src/bridge.rs`. 3 new unit tests + all 6 e2e tests updated and passing with real ECDSA.
- [x] **Multi-RPC fallback.** `EthClient` accepts `Vec<String>` RPC URLs. `rotate_endpoint()` cycles to next URL on failure, tracks consecutive failures per endpoint. `rotate_endpoint(threshold)` returns true when ALL endpoints exceed the threshold — caller can trigger `BridgeEmergencyPause`. Actual pause submission is TODO (requires orchestrator wiring).
- [x] **Retry with backoff.** New `retry.rs` module with `retry_with_backoff()` async utility — exponential backoff (400ms initial, 2x multiplier, 120s max interval, configurable max elapsed time). Only retries on `TransientProviderError` and `ProviderError`. Wired into EthSyncer's finalized block poller and event query paths. Added `_with_retry()` convenience methods to EthClient. 4 unit tests.

#### Integration tests

- [x] **EthClient wiremock tests.** Added `wiremock` dev dependency. 3 tests using mock HTTP server: `test_get_last_finalized_block` (JSON-RPC → block number), `test_get_deposit_events` (ABI-encoded log → parsed DepositEvent), `test_transient_error_detection` (-32005 → TransientProviderError).
- [x] **EthClient unit tests.** `test_deposit_log_parsing` — ABI decoding. `test_wrong_contract_address_returns_none`. `test_short_data_returns_none`. `test_endpoint_rotation` — rotation and failure tracking. Total: 7 EthClient tests.
- [x] **EthSyncer unit tests.** 4 wiremock-based tests: `test_syncer_detects_finalized_block` — full lifecycle with watch channel. `test_syncer_emits_deposit_events` — deposit events emitted via mpsc channel. `test_query_events_with_retry_range_halving` — range > max chunks correctly. `test_query_events_with_retry_transient_error_halves` — -32005 halves range then succeeds.
- [x] **Crypto unit tests.** 8 tests in `types.rs`: sign→ecrecover roundtrip, wrong message detection, build_bridge_signatures format verification (aggregated sig layout + individual ecrecover), known-value encoding tests for deposit/withdrawal/emergency/committee_update (exact byte positions), all-action-types ecrecover roundtrip. Known-value tests produce deterministic keccak256 hashes for Solidity cross-verification.
- [ ] **Full deposit round-trip (Ethereum → Soma).** Test with mock Ethereum provider: deposit event at finalized block → EthSyncer emits event → bridge node signs → gRPC exchange with mock peers → aggregated sigs submitted as `BridgeDeposit` system tx → executor mints USDC.
- [ ] **Full withdrawal round-trip (Soma → Ethereum).** `BridgeWithdraw` tx creates `PendingWithdrawal` → checkpoint watcher detects → signs → gRPC exchange → mock Ethereum contract call with aggregated sigs.
- [ ] **Committee rotation.** Epoch boundary detected → new validator set → sign committee update → verify old committee sigs meet threshold → mock Ethereum `updateCommitteeWithSignatures` call.
- [ ] **Emergency pause/unpause.** Trigger pause (low threshold sigs) → verify bridge rejects ops → unpause (high threshold sigs) → verify bridge resumes.

### Phase 4: Data layer

- [x] **Write diesel migrations.** Fresh-testnet approach: deleted 9 old migration directories, rewrote `soma_epoch_state` in place, created 4 new clean migrations (`soma_asks`, `soma_bids`, `soma_settlements`, `soma_vaults`) with all indexes for reputation graph queries.
- [x] **Create stored types in indexer-alt-schema.** Added `StoredAsk`, `StoredBid`, `StoredSettlement`, `StoredVault` in `soma.rs`. Updated `StoredEpochState` with new marketplace/emission fields. Removed `StoredTarget`, `StoredModel`, `StoredTargetReport`, `StoredReward`, `StoredRewardBalance`.
- [x] **Update `schema.rs`.** Rewrote diesel schema: removed 5 old tables, added 4 new marketplace tables, updated `soma_epoch_state` columns and `allow_tables_to_appear_in_same_query!`.
- [x] **Create indexer pipelines.** Created `soma_asks.rs`, `soma_bids.rs`, `soma_settlements.rs`, `soma_vaults.rs` handlers in `indexer-alt/src/handlers/`. Each scans checkpoint output objects via `Object::deserialize_contents()`, produces stored rows. All registered as Tier C (never pruned). Updated `soma_epoch_state` handler for new fields.
- [x] **Create GraphQL types and queries.** Created `types/ask.rs`, `types/bid.rs`, `types/settlement.rs`, `types/reputation.rs`. Added query resolvers: `asks`, `bids`, `settlements`, `reputation`. Reputation computed via SQL aggregation from settlements + bids tables. Updated `epoch_state` type and query for new schema.
- [x] **Remove old GraphQL types.** Deleted `types/reward.rs`, `types/aggregates.rs`. Removed `TargetReportersLoader` and `TargetRewardLoader` from loaders. Updated all test files.

### Phase 5: Client layer

- [x] **SDK cleanup.** Removed `list_targets()`, `get_architecture_version()`, `score()`, `scoring_health()`, `scoring_url()`, all `scoring` type aliases/imports/fields. `grpc-services` feature now only gates `admin`. Added 6 marketplace transaction builder methods: `create_ask()`, `cancel_ask()`, `create_bid()`, `accept_bid()`, `rate_seller()`, `withdraw_from_vault()`. 24 SDK tests pass.
- [x] **CLI marketplace commands.** Created 5 command modules: `ask` (create with `--task` auto-hash, `--max-price` USDC, `--timeout` human duration; cancel), `bid` (create with `--response` auto-hash, `--price` USDC), `accept` (infers ask-id from bid, `--payment-coin`), `rate` (negative seller rating), `vault` (withdraw with optional amount). Added `usdc_amount.rs` (UsdcAmount type, 6 decimals, microdollars) and `parse_duration_ms()` (30s, 5m, 1h, 1d, 500ms). 25 CLI lib tests + 15 integration tests pass.
- [x] **RPC cleanup.** Removed target/challenge stubs from proto and state_service. Added `GetBid` RPC endpoint (object lookup by ID with field masking). Removed `list_targets()` and `get_architecture_version()` from RPC client.
- [x] **CLI: unified `soma transfer`.** Replaced `send`/`pay` with unified `soma transfer <amount> <recipient>`. Positional amount + recipients for the common case. `--usdc` flag sends USDC instead of SOMA. `--amounts` for per-recipient split. `--coins` for explicit coin selection. Auto-selects coins (SOMA or USDC). USDC transfers auto-select separate SOMA gas coin. Old `send`/`pay` commands hidden but still work for backward compat. Object transfer moved to `soma transfer-object`. `soma merge` (alias: `merge-coins`) unchanged. 25 CLI lib tests + 15 integration tests pass.
- [x] **CLI: `soma accept` auto-USDC selection.** `soma accept <BID_ID>` now auto-selects the richest USDC coin as payment — no `--payment-coin` required. Added `get_richest_usdc_coin()` and `get_usdc_coins_sorted_by_balance()` helpers to `WalletContext` (fetches all coins from RPC, filters client-side by `CoinType::Usdc`, sorts richest-first).
- [x] **CLI: `soma accept --cheapest`.** Fully wired: fetches pending bids via `get_bids_for_ask` RPC, sorts by price ascending, accepts up to `--count N` cheapest sequentially. Auto-selects fresh USDC and gas coins per accept.
- [x] **RPC: marketplace query endpoints.** Added `GetAsk`, `GetSettlement`, `GetBidsForAsk` (with `bids_by_ask` RocksDB secondary index), `GetOpenAsks` (with `open_asks` RocksDB index), `GetVault` (via `owned_objects_iter`), `SubscribeAsks`/`SubscribeBids` (gRPC streaming stubs — empty stream, polling used as v1 fallback). Bumped RPC index DB version to 5.
- [x] **SDK: marketplace query methods.** Added `get_ask()`, `get_bid_object()`, `get_settlement()`, `get_bids_for_ask()`, `get_open_asks()`, `get_vaults()` to both `rpc::Client` and `SomaClient`.
- [x] **CLI: `soma ask list`, `soma ask info`, `soma ask listen`.** Sellers can discover open asks (`list`), inspect details (`info`), and poll for new asks in real-time (`listen` with `--interval` and `--json`).
- [x] **CLI: `soma bid list`, `soma bid listen`.** Buyers can list bids on their ask (`list --ask <ID>` with `--mine`/`--status` filters), and poll for new bids in real-time (`listen <ask-id>` with `--interval` and `--json`).
- [x] **CLI: `soma reputation`, `soma settlements`.** `soma settlements` queries settlements by buyer/seller via new `GetSettlements` RPC endpoint (uses `settlements_by_buyer` and `settlements_by_seller` secondary indexes). `soma reputation` computes buyer/seller reputation from settlement data. Both support `--json`.
- [x] **RPC: `get_settlements` endpoint + secondary indexes.** Added `settlements_by_buyer` and `settlements_by_seller` DBMaps to RpcIndex, populated during checkpoint/tx indexing. `GetSettlements` RPC with buyer/seller filters and page_size. DB version bumped to 6.
- [x] **SDK: `get_settlements()`.** Added to both `rpc::Client` and `SomaClient`.
- [ ] **RPC: real-time streaming.** Wire broadcast channels from checkpoint/tx processing into `SubscribeAsks`/`SubscribeBids` for push-based notifications. Currently return empty streams; CLI uses polling.
- [x] **SDK: remaining query methods.** `get_reputation()`, `get_protocol_fund()` added to both `rpc::Client` and `SomaClient`. CLI `soma reputation` rewired to use server-side GetReputation RPC. 2 new e2e tests validate both endpoints.

### Phase 6: Testing

- [x] **E2E: full happy path.** `test_marketplace_happy_path` — Create ask → create bid → accept bid. Verifies settlement created (buyer/seller/amount/rating), vault credited (net of fee), ask Filled, bid Accepted, USDC deducted from buyer.
- [x] **E2E: multi-bid competition.** `test_multi_bid_competition` — Ask with num_bids_wanted=3, 3 sellers bid at different prices, buyer accepts all 3. Verifies ask Filled with accepted_bid_count=3.
- [x] **E2E: cancellation.** `test_cancel_ask` — Create ask, cancel before accepting. Verifies status Cancelled.
- [x] **E2E: expiry.** `test_ask_expiry` — Create ask with 10s timeout (minimum), advance 3 epochs (5s each) so `epoch_start_timestamp_ms` exceeds ask timeout. Bid on expired ask correctly rejected with `AskExpired`. Note: lazy expiry — ask status remains Open in storage but bids are rejected. Status change to Expired would require explicit sweep or access-time mutation.
- [x] **E2E: negative seller rating.** `test_negative_seller_rating` — Full flow + RateSeller. Verifies settlement.seller_rating = Negative.
- [x] **E2E: default positive rating.** `test_default_positive_rating` — Accept bid with short rating_window_ms (10s), advance 3 epochs past deadline, verify RateSeller rejected with `RatingDeadlinePassed`, settlement.seller_rating remains Positive.
- [x] **E2E: edge cases.** `test_seller_cannot_bid_own_ask` (self-bid rejected), `test_accept_wrong_coin_type` (SOMA instead of USDC rejected), `test_cancel_after_accept_rejected` (cancel after accepting bid rejected).
- [x] **E2E: vault accumulation.** `test_vault_accumulation` — Seller fulfills 2 asks at different prices. Verifies per-settlement vault creation with correct balances, withdraw from both vaults, total USDC matches, vaults deleted after full withdrawal.
- [ ] **E2E: reputation queries (GraphQL).** After mixed positive/negative ratings across multiple counterparties, verify GraphQL reputation endpoint returns correct seller approval rate, bid-to-win ratio, buyer acceptance patterns, counterparty diversity, and rating provenance breakdown (established vs new buyers). (Requires Postgres. Note: RPC-level GetReputation is validated by `test_get_reputation_rpc` e2e test.)
- [x] **E2E: value_fee accounting.** `test_value_fee_accounting` — Verifies settlement amount = bid_price - value_fee (2.5%) and vault balance matches.
- [x] **E2E: CoinType enforcement.** `test_transfer_mixed_coin_types_rejected` (SOMA + USDC in Transfer → rejected), `test_merge_mixed_coin_types_rejected` (SOMA + USDC in MergeCoins → rejected). AcceptBid with SOMA coin covered by `test_accept_wrong_coin_type` above.
- [x] **E2E: unified Transfer.** `test_transfer_soma_single` (single-coin single-recipient SOMA), `test_transfer_usdc_single` (single-coin single-recipient USDC with separate gas), `test_transfer_multi_recipient` (per-recipient amounts split), `test_merge_coins_soma` (merge 2 SOMA coins), `test_merge_coins_usdc` (merge 2 USDC coins with separate gas). All verify correct CoinType and amounts on created/modified coins.
- [x] **E2E: USDC + epoch supply conservation.** `test_usdc_epoch_supply_conservation` — creates cluster with USDC genesis + 5s epochs, runs full marketplace flow (ask→bid→accept), waits for 2 epoch transitions. Validates that `check_soma_conservation()` correctly excludes `CoinType::Usdc` coins from the SOMA supply total. This was the previously-known supply conservation bug, now fixed.
- [x] **E2E: bridge deposit.** `test_bridge_deposit_mints_usdc` — BridgeDeposit with 2/4 committee members mints CoinType::Usdc to recipient. `test_bridge_deposit_insufficient_stake_rejected` — 1/4 members rejected. `test_bridge_deposit_withdraw_roundtrip` — deposit, withdraw, deposit again with new nonce.
- [x] **E2E: bridge withdrawal.** `test_bridge_withdraw_e2e` — User burns USDC via BridgeWithdraw, PendingWithdrawal created with correct sender/amount/eth_address/nonce, USDC coin balance reduced.
- [x] **E2E: bridge emergency pause/unpause.** `test_bridge_emergency_pause_unpause` — Pause (1 member, low threshold), deposit rejected, withdraw rejected, unpause (3 members, high threshold), deposit succeeds.
- [x] **E2E: bridge nonce replay.** `test_bridge_deposit_nonce_replay_rejected` — First deposit succeeds, second with same nonce rejected.
- [ ] **Indexer tests.** All new pipelines (ask, bid, settlement, vault) produce correct DB rows. Settlement rating updates (RateSeller) correctly mutate existing rows. Epoch state reflects new emission model.
- [ ] **GraphQL tests.** Reputation queries with counterparty breakdown, negative rating provenance, bid-to-win ratios. Ask/bid/settlement queries with all filter combinations.
- [x] **E2E: RPC marketplace queries.** `test_rpc_marketplace_queries` — End-to-end ask→bid→accept with object state verification via `get_object` at each step. `test_vault_withdrawal` — Seller withdraws USDC from vault, verifies coin created with correct balance. `test_secondary_index_consistency` — validates marketplace-specific RPC endpoints (GetOpenAsks, GetBidsForAsk with status filter) after field mask fix.
- [x] **E2E: `soma accept --cheapest`.** `test_accept_cheapest` — 3 sellers bid at [1.50, 0.80, 1.20], buyer accepts 2 cheapest via `get_bids_for_ask` + sort. Verifies cheapest 2 Accepted, most expensive still Pending, status filter returns 1 remaining.
- [ ] **E2E: seller listen flow.** Seller runs `soma ask listen --json`, buyer creates ask, verify the ask appears in seller's output. Seller bids, buyer accepts, verify settlement.
- [ ] **E2E: buyer listen flow.** Buyer creates ask, runs `soma bid listen <ask-id> --json`, seller creates bid, verify the bid appears in buyer's output. Buyer accepts.
- [x] **E2E: secondary index consistency.** `test_secondary_index_consistency` — Creates 2 asks, verifies `open_asks` index. Bids on ask[0], verifies `bids_by_ask`. Accepts → ask[0] removed from open_asks. Status filter (Pending/Accepted) verified. Cancel ask[1] → removed from open_asks.
- [ ] **E2E: streaming RPCs (when wired).** Subscribe to asks stream, create ask, verify event received. Subscribe to bids stream for an ask, create bid, verify event received. Verify event_type field is correct ("created", "filled", etc.).

### Phase 7: Deploy

- [ ] Deploy Ethereum bridge contract (testnet Ethereum Sepolia or mainnet depending on stage)
- [ ] Fresh testnet genesis with marketplace parameters, bridge committee, SOMA allocation to validators
- [ ] Verify SOMA faucet distributes gas tokens
- [ ] Deploy bridge nodes alongside each validator
- [ ] Update K8s manifests (validators + bridge nodes + indexer + graphql)
- [ ] Run DB migrations on indexer Postgres
- [ ] Deploy and verify indexer + GraphQL
- [ ] Verify Protocol Fund accumulates USDC on AcceptBid transactions
- [ ] Deploy Glass server (separate repo)
- [ ] Deploy MCP server (separate repo)
- [ ] Smoke test bridge: deposit USDC on Ethereum, verify minted on Soma, withdraw back
- [ ] Smoke test full flow through MCP → Glass server → chain → indexer → GraphQL

---

## What Does Not Change

- **Consensus**: Mysticeti — identical
- **Validator infrastructure**: staking, commission, set management, epoch rotation
- **Object model**: BCS-serialized Rust structs, same storage layer
- **Transaction framework**: gas preparation, fee deduction, temporary store, effects generation
- **Checkpoint system**: creation, sync, ingestion
- **Indexer framework**: generic pipeline architecture
- **Cryptography**: Ed25519 signing, BLS for validators. New: Secp256k1 bridge keys per validator (EVM-compatible).
- **SOMA token**: **1B SOMA** (changed from 10M), shannons denomination (1B shannons per SOMA). Emission schedule rewritten to geometric step-decay (Sui's StakeSubsidy model). CoinType tag distinguishes SOMA from USDC coins — same Coin storage and methods.
- **Coin operations**: Unified into `Transfer` (multi-coin input, multi-recipient output) and `MergeCoins` (combine N coins into one). Replaces old `TransferCoin`/`PayCoins` split.
- **P2P/networking**: peer discovery, sync protocols
- **K8s deployment topology**: fullnode + indexer + graphql. New: bridge node sidecar per validator.
- **Client interface**: CLI only. Python SDK removed — agents and humans both use `soma` CLI (with `--json` for structured output). No PyO3/maturin build complexity.
