# Soma Monorepo Architecture Guide

Comprehensive reference for developers working on the Soma blockchain — a BFT consensus chain with a continuous mining competition where data miners race to fill targets using registered ML models.

## Table of Contents
1. [System Overview](#system-overview)
2. [Crate Organization](#crate-organization)
3. [Core Data Model](#core-data-model)
4. [Transaction Types](#transaction-types)
5. [Execution Engine](#execution-engine)
6. [Epoch Transitions & Safe Mode](#epoch-transitions--safe-mode)
7. [Mining Competition Flow](#mining-competition-flow)
8. [Key Architectural Patterns](#key-architectural-patterns)
9. [RPC & CLI](#rpc--cli)
10. [Testing Infrastructure](#testing-infrastructure)
11. [Build Commands](#build-commands)
12. [File Reference](#file-reference)
13. [Glossary](#glossary)

---

## System Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                                 │
│              CLI / Rust SDK / Python SDK / RPC (gRPC + JSON)          │
└───────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        FULLNODE LAYER                                  │
│  ┌──────────────────────┐  ┌──────────────────────────────────────┐   │
│  │ Transaction           │  │ Authority State                      │   │
│  │ Orchestrator          │  │ (execution engine + object cache)    │   │
│  │ (submit + finality)   │  │                                      │   │
│  └──────────────────────┘  └──────────────────────────────────────┘   │
│            │                              │                            │
│            ▼                              ▼                            │
│  ┌──────────────────────┐  ┌──────────────────────────────────────┐   │
│  │ Consensus Manager     │  │ Checkpoint Service                   │   │
│  │ (Mysticeti BFT)       │  │ (finality + state sync)              │   │
│  └──────────────────────┘  └──────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
```

The system implements a **continuous mining competition**:

1. **Model owners** register ML models via commit-reveal (commit epoch N, reveal epoch N+1)
2. **Targets** are generated at epoch boundaries with stake-weighted model selection and difficulty-controlled radius
3. **Miners** find data that embeds within a target's radius, submit on-chain with a bond
4. **Challenges** allow anyone to dispute a submission; validators audit and vote (2f+1 quorum resolves)
5. **Rewards** are distributed at claim time: miner (50%) + model owner (30%) + claimer incentive (1%), with validators receiving their share from epoch emissions

---

## Crate Organization

### Workspace (25 crates)

| Category | Crate | Purpose |
|----------|-------|---------|
| **Types & Config** | `types` | All shared data structures, transactions, objects, crypto |
| | `protocol-config` | Protocol version configuration and system parameters |
| | `protocol-config-macros` | Procedural macros for protocol config |
| **Consensus** | `consensus` | Mysticeti BFT consensus implementation |
| | `authority` | Validator state, execution engine, all transaction executors |
| | `node` | Fullnode/validator orchestration (starts all components) |
| | `runtime` | Runtime utilities |
| **Storage** | `store` | RocksDB-backed persistent storage |
| | `store-derive` | Derive macros for storage traits |
| | `blobs` | Object storage abstraction (S3, GCS, Azure, HTTP) |
| **Networking** | `sync` | P2P state synchronization |
| | `soma-http` | HTTP client/server layer |
| | `soma-tls` | TLS certificate management |
| | `soma-keys` | Cryptographic key management |
| | `rpc` | gRPC service definitions, proto codegen, client SDK |
| **User Interface** | `cli` | Command-line interface (`soma` binary) |
| | `sdk` | Rust client SDK |
| | `python-sdk` | Python bindings (PyO3) |
| **ML** | `models` | Model-related utilities |
| **Utilities** | `utils` | Shared utilities (compression, logging, codec, failpoints) |
| | `data-ingestion` | Data ingestion pipelines |
| | `arrgen` | Deterministic array generation for testing |
| | `bin-version` | Binary version information |
| **Testing** | `test-cluster` | Validator cluster simulation (msim) |
| | `e2e-tests` | End-to-end integration tests (93 tests) |

### Dependency Graph

```
                    ┌──────────────┐
                    │    types     │ ◄─── All crates depend on types
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐      ┌──────────┐
   │  store   │     │  utils   │      │   rpc    │
   └────┬─────┘     └──────────┘      └────┬─────┘
        │                                  │
        └──────────────┬───────────────────┘
                       │
                       ▼
                 ┌──────────┐
                 │authority │
                 └────┬─────┘
                      │
                      ▼
                 ┌──────────┐
                 │   node   │
                 └────┬─────┘
                      │
                      ▼
                 ┌──────────┐
                 │   cli    │
                 └──────────┘
```

---

## Core Data Model

### Object Model

Every piece of on-chain state is an **Object**:

```rust
Object
├── ObjectID (32 bytes, same type as SomaAddress)
├── Version (u64, lamport timestamp)
├── Digest (SHA256 of contents)
├── Owner
│   ├── AddressOwner(SomaAddress)           # Exclusively owned, mutable
│   ├── Shared { initial_shared_version }   # Consensus-sequenced, mutable by anyone
│   └── Immutable                           # Read-only
└── Data (BCS-serialized contents)
```

### Object Types

| ObjectType | Owner | Description |
|------------|-------|-------------|
| `SystemState` | Shared | Global blockchain state (one per chain) |
| `Coin` | AddressOwner | SOMA token balance |
| `StakedSoma` | AddressOwner | Staked tokens (validator or model) |
| `Target` | Shared | Mining target with embedding, radius, reward pool |
| `Submission` | Shared | Data submission to a target (audit data for challenges) |
| `Challenge` | Shared | Fraud dispute against a submission |

### SystemState

The single most important object — contains all global chain state:

```rust
pub struct SystemState {
    pub epoch: u64,
    pub protocol_version: u64,
    pub validators: ValidatorSet,
    pub parameters: SystemParameters,
    pub epoch_start_timestamp_ms: u64,
    pub validator_report_records: BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
    pub model_registry: ModelRegistry,
    pub emission_pool: EmissionPool,
    pub target_state: TargetState,

    // Safe mode fields (liveness guarantee)
    pub safe_mode: bool,
    pub safe_mode_accumulated_fees: u64,
    pub safe_mode_accumulated_emissions: u64,
}
```

**Key sub-structures:**

- **`ModelRegistry`** — Active/pending/inactive models, staking pool mappings, total model stake, report records
- **`EmissionPool`** — Remaining balance from genesis + fixed `emission_per_epoch`
- **`TargetState`** — Current `distance_threshold`, `targets_generated_this_epoch`, `hits_this_epoch`, `hit_rate_ema_bps`, `reward_per_target`
- **`SystemParameters`** — All protocol-tunable constants (40+ fields covering epochs, fees, models, targets, rewards, bonds, challenges)

### Target

The core mining primitive:

```rust
pub struct Target {
    pub embedding: SomaTensor,              // Center point miners must match
    pub model_ids: Vec<ModelId>,            // Assigned models (stake-weighted selection)
    pub distance_threshold: SomaTensor,     // Max cosine distance for valid submission
    pub reward_pool: u64,                   // Pre-allocated from emission pool
    pub generation_epoch: EpochId,
    pub status: TargetStatus,               // Open | Filled { fill_epoch } | Claimed
    pub miner: Option<SomaAddress>,         // Who filled it
    pub winning_model_id: Option<ModelId>,
    pub winning_model_owner: Option<SomaAddress>,
    pub bond_amount: u64,                   // Miner's bond held until claim
    pub winning_data_manifest: Option<SubmissionManifest>,
    pub winning_data_commitment: Option<DataCommitment>,
    pub winning_embedding: Option<SomaTensor>,
    pub winning_distance_score: Option<SomaTensor>,
    pub challenger: Option<SomaAddress>,    // First challenger wins slot
    pub challenge_id: Option<ChallengeId>,
    pub submission_reports: BTreeMap<SomaAddress, Option<SomaAddress>>,  // Tally-based fraud detection
}
```

### Challenge

```rust
pub struct Challenge {
    pub id: ChallengeId,
    pub target_id: TargetId,
    pub challenger: SomaAddress,
    pub challenger_bond: u64,
    pub challenge_epoch: EpochId,
    pub status: ChallengeStatus,            // Pending | Resolved { challenger_lost }
    // Audit data (copied from Target at creation)
    pub model_ids: Vec<ModelId>,
    pub target_embedding: SomaTensor,
    pub distance_threshold: SomaTensor,
    pub winning_model_id: ModelId,
    pub winning_data_manifest: SubmissionManifest,
    pub winning_data_commitment: DataCommitment,
    pub winning_embedding: SomaTensor,
    pub winning_distance_score: SomaTensor,
    pub challenge_reports: BTreeSet<SomaAddress>,  // Validators who say challenger is wrong
}
```

### Model

```rust
pub struct Model {
    pub model_id: ModelId,
    pub owner: SomaAddress,
    pub weights_manifest: ModelWeightsManifest,
    pub weights_url_commitment: ModelWeightsUrlCommitment,
    pub weights_commitment: ModelWeightsCommitment,
    pub architecture_version: ArchitectureVersion,
    pub staking_pool: StakingPool,
    pub commission_rate: u64,               // BPS
    pub next_epoch_commission_rate: u64,
    pub pending_update: Option<PendingModelUpdate>,
    // ... lifecycle fields
}
```

---

## Transaction Types

31 transaction variants in [types/src/transaction.rs](types/src/transaction.rs):

### System (3)
| Transaction | Description |
|-------------|-------------|
| `Genesis` | Initialize blockchain state |
| `ConsensusCommitPrologue` | Consensus round marker |
| `ChangeEpoch` | Epoch transition (reward distribution, model processing, target generation, safe mode fallback) |

### Validator Management (6)
| Transaction | Description |
|-------------|-------------|
| `AddValidator` | Add validator to the set |
| `RemoveValidator` | Remove validator |
| `UpdateValidatorMetadata` | Update validator info |
| `SetCommissionRate` | Set validator commission (BPS) |
| `ReportValidator` | Report validator misbehavior |
| `UndoReportValidator` | Retract validator report |

### Coin & Object Operations (3)
| Transaction | Description |
|-------------|-------------|
| `TransferCoin` | Send SOMA (with optional partial amount) |
| `PayCoins` | Multi-recipient payment |
| `TransferObjects` | Transfer arbitrary objects |

### Staking (2)
| Transaction | Description |
|-------------|-------------|
| `AddStake` | Stake with a validator |
| `WithdrawStake` | Withdraw staked SOMA |

### Model Management (9)
| Transaction | Description |
|-------------|-------------|
| `CommitModel` | Register new model (commit phase) |
| `RevealModel` | Reveal model weights (next epoch) |
| `CommitModelUpdate` | Commit weight update for active model |
| `RevealModelUpdate` | Reveal updated weights |
| `AddStakeToModel` | Stake tokens to model's staking pool |
| `SetModelCommissionRate` | Set model commission rate |
| `DeactivateModel` | Voluntarily deactivate model |
| `ReportModel` | Validator reports model unavailability |
| `UndoReportModel` | Retract model report |

### Submission (4)
| Transaction | Description |
|-------------|-------------|
| `SubmitData` | Submit data to fill an open target (locks bond) |
| `ClaimRewards` | Claim rewards from filled target or return expired target to pool |
| `ReportSubmission` | Validator reports fraudulent submission (tally on Target) |
| `UndoReportSubmission` | Retract submission report |

### Challenge (4)
| Transaction | Description |
|-------------|-------------|
| `InitiateChallenge` | Challenge a filled target (locks challenger bond) |
| `ReportChallenge` | Validator says challenger is wrong (tally on Challenge) |
| `UndoReportChallenge` | Retract challenge report |
| `ClaimChallengeBond` | Resolve challenge and distribute bond |

---

## Execution Engine

### Pipeline

Located in [authority/src/execution/mod.rs](authority/src/execution/mod.rs):

```
Transaction Received
        │
        ▼
┌───────────────────────────────────────────┐
│  1. Executor Dispatch                     │
│     create_executor(kind) → specialized   │
│     executor for this transaction type    │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  2. Gas Preparation                       │
│     - Smash gas coins                     │
│     - Deduct base fee (DOS protection)    │
│     - System txs skip gas handling        │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  3. Input Loading                         │
│     - Owned objects by ObjectRef           │
│     - Shared objects with assigned versions│
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  4. Execute                               │
│     executor.execute(&mut store, ...)     │
│     On error: revert non-gas changes      │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  5. Value Fees + Ownership Invariants     │
│     - Deduct operation-specific fees      │
│     - Verify mutable inputs still valid   │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  6. Generate Effects                      │
│     - Created/mutated/deleted objects     │
│     - Gas cost breakdown                  │
│     - Execution status                    │
└───────────────────────────────────────────┘
```

### Executor Dispatch

| Executor | File | Transactions |
|----------|------|-------------|
| `ValidatorExecutor` | [execution/validator.rs](authority/src/execution/validator.rs) | AddValidator, RemoveValidator, ReportValidator, UndoReportValidator, SetCommissionRate, UpdateValidatorMetadata |
| `CoinExecutor` | [execution/coin.rs](authority/src/execution/coin.rs) | TransferCoin, PayCoins |
| `ObjectExecutor` | [execution/object.rs](authority/src/execution/object.rs) | TransferObjects |
| `StakingExecutor` | [execution/staking.rs](authority/src/execution/staking.rs) | AddStake, WithdrawStake |
| `ModelExecutor` | [execution/model.rs](authority/src/execution/model.rs) | 9 model transactions |
| `SubmissionExecutor` | [execution/submission.rs](authority/src/execution/submission.rs) | SubmitData, ClaimRewards, ReportSubmission, UndoReportSubmission |
| `ChallengeExecutor` | [execution/challenge.rs](authority/src/execution/challenge.rs) | InitiateChallenge, ReportChallenge, UndoReportChallenge, ClaimChallengeBond |
| `ChangeEpochExecutor` | [execution/change_epoch.rs](authority/src/execution/change_epoch.rs) | ChangeEpoch |
| `GenesisExecutor` | [execution/system.rs](authority/src/execution/system.rs) | Genesis |
| `ConsensusCommitExecutor` | [execution/system.rs](authority/src/execution/system.rs) | ConsensusCommitPrologue |

### Consensus Integration

Located in [authority/src/consensus_handler.rs](authority/src/consensus_handler.rs):

```
Mysticeti Consensus Output
        │
        ▼
ConsensusHandler::handle_consensus_commit()
        │
        ├─ Filter & deduplicate transactions
        ├─ Assign shared object versions
        ├─ Create ConsensusCommitPrologue
        │
        ▼
ExecutionScheduler → ExecutionDriver
        │
        └─ AuthorityState::try_execute_immediately()
```

---

## Epoch Transitions & Safe Mode

### Normal Epoch Flow

Located in [authority/src/execution/change_epoch.rs](authority/src/execution/change_epoch.rs) and [types/src/system_state/mod.rs](types/src/system_state/mod.rs):

```
End of Epoch Detected
        │
        ▼
ChangeEpochExecutor::execute()
        │
        ├─ Clone state_backup
        ├─ Call state.advance_epoch()
        │
        ▼ (On success)
advance_epoch():
  ├─ Drain safe mode accumulators (if recovering)
  ├─ Calculate total_rewards = fees + emissions + safe_mode_extra
  ├─ Allocate validator rewards (validator_reward_allocation_bps)
  ├─ Adjust difficulty (hit rate EMA → distance threshold)
  ├─ Recalculate reward_per_target
  ├─ Process validator rewards (slashing for reported validators)
  ├─ Process model registry (report quorum → slash, reveal timeouts, commission updates, staking pools)
  └─ Return validator_rewards map

Then in executor:
  ├─ Create StakedSoma reward objects for validators
  ├─ Generate initial targets for new epoch (stake-weighted model selection)
  └─ Serialize & commit SystemState
```

### Safe Mode (Liveness Guarantee)

If `advance_epoch()` fails for any reason, the network enters **safe mode** instead of crashing:

```
advance_epoch() returns Err
        │
        ▼
Restore state from state_backup
        │
        ▼
advance_epoch_safe_mode():
  ├─ Set safe_mode = true
  ├─ Increment epoch + timestamp
  ├─ Accumulate fees → safe_mode_accumulated_fees
  ├─ Accumulate emissions → safe_mode_accumulated_emissions
  └─ Skip everything else (rewards, models, targets, difficulty)

Network continues in degraded mode:
  ├─ User transactions still work
  ├─ No validator rewards distributed
  ├─ Model registry frozen
  ├─ No new targets generated
  └─ Fix deployed via protocol upgrade; next successful advance_epoch drains accumulators
```

The fatal `assert!(effects.status().is_ok())` in authority.rs has been replaced with an `error!()` log, providing defense-in-depth.

### Arithmetic Hardening

All BPS calculations in epoch transitions use u128 intermediates to prevent overflow:
```rust
let allocation = (total_rewards as u128 * allocation_bps as u128 / BPS_DENOMINATOR as u128) as u64;
```

---

## Mining Competition Flow

### Target Lifecycle

```
1. GENERATION (epoch boundary in ChangeEpochExecutor)
   ├─ Seed = hash(tx_digest, creation_num)
   ├─ Embedding = deterministic_embedding(seed, embedding_dim)
   ├─ Models = stake-weighted KNN selection from active models
   ├─ Distance threshold from TargetState
   ├─ Reward pre-allocated from emission pool
   └─ Status: Open

2. SUBMISSION (SubmitData transaction)
   ├─ Miner provides: data manifest, commitment, embedding, distance score, model choice
   ├─ Validates: target is Open, model is in target's list, distance ≤ threshold
   ├─ Bond locked: submission_bond_per_byte × data_size
   ├─ Target status → Filled { fill_epoch }
   ├─ Creates Submission shared object with audit data
   └─ Spawns replacement target (if emission pool has funds)

3. CHALLENGE WINDOW (fill_epoch only)
   ├─ Anyone can InitiateChallenge (locks challenger_bond_per_byte × data_size)
   ├─ Validators audit: download data, verify embedding, submit ReportChallenge if challenger is wrong
   └─ First challenger wins the slot

4. CLAIM (ClaimRewards, after challenge window closes)
   ├─ current_epoch > fill_epoch + 1 required
   ├─ Check submission_reports for 2f+1 quorum
   │
   ├─ NO fraud quorum (normal):
   │   ├─ Miner gets target_miner_reward_share_bps (50%)
   │   ├─ Model owner gets target_model_reward_share_bps (30%)
   │   ├─ Claimer gets target_claimer_incentive_bps (1%)
   │   ├─ Remainder → emission pool
   │   └─ Bond returned to miner
   │
   └─ FRAUD quorum:
       ├─ Bond → challenger (or split among reporting validators)
       └─ Rewards → emission pool

5. EXPIRED (unfilled target past its epoch)
   ├─ ClaimRewards returns most of reward_pool to emission pool
   └─ Claimer receives small incentive for cleanup
```

### Challenge Resolution

```
InitiateChallenge → Creates Challenge shared object
        │
        ▼
Validators audit the submission data
        │
        ├─ If submission IS valid → ReportChallenge (challenger is wrong)
        │
        ▼
ClaimChallengeBond (after challenge_epoch)
        │
        ├─ 2f+1 reports → Challenger loses, bond → reporting validators
        └─ No quorum → Challenger wins (benefit of doubt), bond returned
```

### Difficulty Adjustment

At each epoch boundary in `advance_epoch_targets()`:
- Calculate `hit_rate = hits_this_epoch / targets_generated_this_epoch`
- Update `hit_rate_ema_bps` with exponential decay
- If EMA > target: decrease distance_threshold (harder)
- If EMA < target: increase distance_threshold (easier)
- Clamp between `min_distance_threshold` and `max_distance_threshold`
- Recalculate `reward_per_target` from emissions allocation

---

## Key Architectural Patterns

### 1. Envelope Pattern (Signatures)

```rust
struct Envelope<T: Message, S: Signature> { data: T, auth_signature: S }

type CertifiedTransaction = Envelope<SenderSignedData, AuthorityStrongQuorumSignInfo>;
type CertifiedCheckpoint = Envelope<CheckpointSummary, AuthorityStrongQuorumSignInfo>;
```

### 2. Per-Epoch State Isolation

```rust
struct AuthorityState {
    epoch_store: ArcSwap<AuthorityPerEpochStore>,  // Swapped at epoch boundary
    execution_lock: RwLock<EpochId>,               // Read during execution
}
```

### 3. Tally-Based Reporting

Both submissions and challenges use the same pattern — reports accumulate directly on shared objects:
- Validators call `ReportSubmission`/`ReportChallenge` to add their address to a set
- At claim time, `reports.len() >= quorum_threshold` triggers slashing/bond forfeiture
- No separate voting round — reports checked lazily at resolution

### 4. Spawn-on-Fill (Target Replacement)

When a target is filled via `SubmitData`, the executor immediately spawns a replacement target (if the emission pool has sufficient funds). This maintains a constant supply of open targets.

### 5. Bond Economics

Both miners and challengers lock bonds proportional to data size:
- **Miner bond**: `submission_bond_per_byte × data_size` — returned on successful claim, forfeited on fraud
- **Challenger bond**: `challenger_bond_per_byte × data_size` — returned if challenger wins (or no quorum), forfeited to validators if 2f+1 say challenger is wrong

### 6. BPS Convention

All percentages use basis points (1 BPS = 0.01%, 10000 BPS = 100%). Constant: `BPS_DENOMINATOR = 10000`.

### 7. FullnodeConfigBuilder (Test Infrastructure)

Fullnodes in test clusters are created independently from validators:
```rust
let fullnode_config = FullnodeConfigBuilder::new()
    .with_config_directory(dir)
    .build(genesis, seed_peers);
// consensus_config == None → node runs as fullnode
```

### 8. Finality Proofs

```rust
struct FinalityProof {
    transaction: Transaction,
    effects: TransactionEffects,
    checkpoint: CertifiedCheckpointSummary,
    inclusion_proof: CheckpointInclusionProof,  // Merkle proof
}
```

---

## RPC & CLI

### RPC Services

3 gRPC services defined in [rpc/proto/soma/rpc/](rpc/proto/soma/rpc/):

**LedgerService** — Chain data queries:
- `GetServiceInfo`, `GetObject`, `BatchGetObjects`, `GetTransaction`, `BatchGetTransactions`, `GetCheckpoint`, `GetEpoch`

**StateService** — State queries with filtering:
- `ListOwnedObjects`, `GetBalance`, `GetTarget`, `ListTargets`, `GetChallenge`, `ListChallenges`

**TransactionExecutionService** — Submit transactions:
- `ExecuteTransaction`, `SimulateTransaction`

### CLI Commands

Located in [cli/src/](cli/src/) — 17 command files:

**User Operations:**
```bash
soma balance [ADDRESS]                      # Check balance
soma send --to <ADDR> --amount <N>          # Transfer SOMA
soma transfer --to <ADDR> --object-id <ID>  # Transfer object
soma pay --recipients <A1,A2> --amounts <N1,N2>  # Multi-pay
soma stake --validator <ADDR>               # Stake with validator
soma stake --model <MODEL_ID>               # Stake with model
soma unstake <STAKED_ID>                    # Withdraw stake
```

**Model Operations:**
```bash
soma model commit --model-id <ID> --weights-url-commitment <HASH> --weights-commitment <HASH> --architecture-version <N> --stake-amount <N>
soma model reveal --model-id <ID> --weights-url <URL> --weights-checksum <HASH> --weights-size <N> --decryption-key <KEY>
soma model update-commit ...                # Commit weight update
soma model update-reveal ...                # Reveal weight update
soma model deactivate --model-id <ID>
soma model set-commission-rate --model-id <ID> --rate <BPS>
soma model info <ID>
soma model list
```

**Target & Mining Operations:**
```bash
soma target list [--status open|filled|claimed] [--epoch N]
soma target info <TARGET_ID>
soma submit --target-id <ID> --data-url <URL> --model-id <ID> --embedding <CSV> --distance-score <F32> --bond-coin <COIN_ID> ...
soma claim --target-id <ID>
soma data --target-id <ID>                  # Download submission data
```

**Challenge Operations:**
```bash
soma challenge initiate --target-id <ID> --bond-coin <COIN_ID>
soma challenge info <CHALLENGE_ID>
soma challenge list [--target-id <ID>] [--status pending|resolved]
```

**Query Commands:**
```bash
soma objects get <ID>
soma objects list [OWNER]
soma tx <DIGEST>
```

**Operator Commands:**
```bash
soma validator start --config <PATH>
soma validator join-committee <INFO_PATH>
soma validator leave-committee
soma validator report-model <MODEL_ID>
```

---

## Testing Infrastructure

### E2E Tests (93 tests across 14 files)

All E2E tests use the msim deterministic simulator and run with `RUSTFLAGS="--cfg msim"`.

| File | Tests | Coverage |
|------|-------|----------|
| [challenge_tests.rs](e2e-tests/tests/challenge_tests.rs) | 14 | Challenge initiation, bond locking, tally-based resolution, quorum behavior |
| [reconfiguration_tests.rs](e2e-tests/tests/reconfiguration_tests.rs) | 13 | Epoch transitions, validator join/leave, stake-weighted power, reporting |
| [failpoint_tests.rs](e2e-tests/tests/failpoint_tests.rs) | 13 | Crash recovery (6 crash points), delays, safe mode (3 tests), race conditions |
| [transaction_orchestrator_tests.rs](e2e-tests/tests/transaction_orchestrator_tests.rs) | 8 | Submission, WAL persistence, epoch transitions, stale object handling |
| [shared_object_tests.rs](e2e-tests/tests/shared_object_tests.rs) | 8 | Target/SystemState mutations, racing miners, version tracking, replay idempotency |
| [full_node_tests.rs](e2e-tests/tests/full_node_tests.rs) | 7 | Fullnode state sync, orchestrator, RunWithRange, BCS serialization |
| [rpc_tests.rs](e2e-tests/tests/rpc_tests.rs) | 7 | gRPC endpoints, system state queries, balance, object listing |
| [target_tests.rs](e2e-tests/tests/target_tests.rs) | 6 | Target generation, submission, challenge window, expiration, epoch boundary |
| [protocol_version_tests.rs](e2e-tests/tests/protocol_version_tests.rs) | 6 | Protocol upgrades, 2f+1 quorum, laggard shutdown |
| [simulator_tests.rs](e2e-tests/tests/simulator_tests.rs) | 5 | Deterministic msim behavior, ordering, HashMap iteration |
| [model_tests.rs](e2e-tests/tests/model_tests.rs) | 2 | Model commit-reveal lifecycle, staking |
| [checkpoint_tests.rs](e2e-tests/tests/checkpoint_tests.rs) | 2 | Checkpoint creation, timestamp monotonicity |
| [multisig_tests.rs](e2e-tests/tests/multisig_tests.rs) | 2 | MultiSig Ed25519, weighted thresholds |

### TestClusterBuilder

Located in [test-cluster/src/lib.rs](test-cluster/src/lib.rs):

```rust
let test_cluster = TestClusterBuilder::new()
    .with_num_validators(4)
    .with_epoch_duration_ms(5000)
    .with_genesis_models(vec![model_config])  // Seed models
    .build()
    .await;

// Execute transactions
let response = test_cluster.sign_and_execute_transaction(&tx_data).await;

// Wait for epochs
let state = test_cluster.wait_for_epoch_with_timeout(Some(3), Duration::from_secs(60)).await;

// Access node state
test_cluster.fullnode_handle.soma_node.with(|node| { node.state()... });
```

### Failpoint Infrastructure

Located in [utils/src/fp.rs](utils/src/fp.rs), macros in [utils/src/lib.rs](utils/src/lib.rs):

```rust
// Registration (from tests)
register_fail_point("tag", || { ... });
register_fail_point_async("tag", || async { ... });
register_fail_point_if("tag", || -> bool { ... });

// Usage (in production code, compiles to no-op without cfg(msim))
fail_point!("tag");
fail_point_async!("tag");
fail_point_if!("tag", || { ... });
```

**Key patterns:**
- Under `cfg(msim)`: `thread_local!` — all simulated nodes share one OS thread, so test-registered failpoints are visible to validators
- **Crash simulation**: Use `msim::task::kill_current_node(Some(Duration))` NOT `panic!()` — panics fail the entire test in msim
- **Boolean flags over counters**: All validators call failpoint callbacks independently. Counters increment N times per epoch, causing split state. Boolean flags ensure all validators see the same result.

**Active failpoint tags:**

| Tag | Location | Purpose |
|-----|----------|---------|
| `reconfig_delay` | node/src/lib.rs | Delay epoch reconfiguration |
| `change_epoch_tx_delay` | authority/src/authority.rs | Delay ChangeEpoch tx creation |
| `before-open-new-epoch-store` | node/src/lib.rs | Crash before new epoch store |
| `crash-before-commit-certificate` | authority/src/authority.rs | Crash before cert commit |
| `crash-after-consensus-commit` | authority/src/consensus_handler.rs | Crash after consensus |
| `crash-after-accumulate-epoch` | authority/src/checkpoints/ | Crash after epoch accumulation |
| `crash-after-db-write` | authority/src/authority_store.rs | Crash after DB write |
| `crash-after-build-batch` | authority/src/authority_store.rs | Crash after batch build |
| `highest-executed-checkpoint` | authority/src/checkpoints/ | Checkpoint execution observation |
| `advance_epoch_result_injection` | authority/src/execution/change_epoch.rs | Inject epoch failure (safe mode) |
| `checkpoint_builder_advance_epoch_is_safe_mode` | authority/src/authority.rs | Detect unexpected safe mode |

---

## Build Commands

```bash
# Build all crates (PYO3_PYTHON needed for python-sdk in workspace)
PYO3_PYTHON=python3 cargo build --release

# Run unit tests
cargo test -p types
cargo test -p authority

# Build for msim (required for test-cluster and e2e-tests)
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo build -p e2e-tests -p test-cluster

# Run specific E2E test
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test failpoint_tests

# Check a crate under msim without building tests
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo check -p authority

# Start local network
cargo run --bin soma -- start --force-regenesis
```

---

## File Reference

### Core Type Definitions

| File | Description |
|------|-------------|
| [types/src/transaction.rs](types/src/transaction.rs) | 31 transaction variants + args structs |
| [types/src/object.rs](types/src/object.rs) | Object, ObjectID, ObjectType, Owner |
| [types/src/system_state/mod.rs](types/src/system_state/mod.rs) | SystemState + advance_epoch + safe mode |
| [types/src/system_state/model_registry.rs](types/src/system_state/model_registry.rs) | ModelRegistry (active/pending/inactive) |
| [types/src/system_state/target_state.rs](types/src/system_state/target_state.rs) | TargetState (difficulty, hit rate EMA) |
| [types/src/system_state/emission.rs](types/src/system_state/emission.rs) | EmissionPool |
| [types/src/system_state/validator.rs](types/src/system_state/validator.rs) | Validator, ValidatorSet |
| [types/src/system_state/staking.rs](types/src/system_state/staking.rs) | StakingPool, StakedSoma |
| [types/src/model.rs](types/src/model.rs) | Model, ModelId, PendingModelUpdate |
| [types/src/model_selection.rs](types/src/model_selection.rs) | Stake-weighted model selection for targets |
| [types/src/target.rs](types/src/target.rs) | Target, TargetStatus, generate_target(), deterministic_embedding() |
| [types/src/submission.rs](types/src/submission.rs) | Submission, SubmissionManifest |
| [types/src/challenge.rs](types/src/challenge.rs) | Challenge, ChallengeStatus |
| [types/src/tensor.rs](types/src/tensor.rs) | SomaTensor (f32 embedding vectors) |
| [types/src/finality.rs](types/src/finality.rs) | FinalityProof, Merkle inclusion proofs |

### Execution Engine

| File | Description |
|------|-------------|
| [authority/src/execution/mod.rs](authority/src/execution/mod.rs) | Execution pipeline + executor dispatch |
| [authority/src/execution/prepare_gas.rs](authority/src/execution/prepare_gas.rs) | Gas preparation (smash, base fee, value fee) |
| [authority/src/execution/change_epoch.rs](authority/src/execution/change_epoch.rs) | Epoch transition executor (safe mode fallback) |
| [authority/src/execution/submission.rs](authority/src/execution/submission.rs) | SubmitData, ClaimRewards, ReportSubmission |
| [authority/src/execution/challenge.rs](authority/src/execution/challenge.rs) | InitiateChallenge, ReportChallenge, ClaimChallengeBond |
| [authority/src/execution/model.rs](authority/src/execution/model.rs) | 9 model management transactions |
| [authority/src/execution/validator.rs](authority/src/execution/validator.rs) | Validator management |
| [authority/src/execution/coin.rs](authority/src/execution/coin.rs) | TransferCoin, PayCoins |
| [authority/src/execution/staking.rs](authority/src/execution/staking.rs) | AddStake, WithdrawStake |
| [authority/src/execution/object.rs](authority/src/execution/object.rs) | TransferObjects |
| [authority/src/execution/system.rs](authority/src/execution/system.rs) | Genesis, ConsensusCommitPrologue |

### Node & Consensus

| File | Description |
|------|-------------|
| [node/src/lib.rs](node/src/lib.rs) | Fullnode/validator startup + reconfiguration monitor |
| [authority/src/authority.rs](authority/src/authority.rs) | AuthorityState (2300+ lines) |
| [authority/src/authority_server.rs](authority/src/authority_server.rs) | gRPC service handlers |
| [authority/src/consensus_handler.rs](authority/src/consensus_handler.rs) | Consensus commit handling |

### Configuration & Genesis

| File | Description |
|------|-------------|
| [types/src/config/node_config.rs](types/src/config/node_config.rs) | ValidatorConfigBuilder + FullnodeConfigBuilder |
| [types/src/config/network_config.rs](types/src/config/network_config.rs) | Network/committee configuration |
| [types/src/config/genesis_config.rs](types/src/config/genesis_config.rs) | Genesis config (validators, models, accounts) |
| [types/src/genesis_builder.rs](types/src/genesis_builder.rs) | Genesis state creation (validators + seed models + initial targets) |
| [protocol-config/src/lib.rs](protocol-config/src/lib.rs) | ProtocolConfig with 40+ versioned parameters |

### RPC

| File | Description |
|------|-------------|
| [rpc/proto/soma/rpc/](rpc/proto/soma/rpc/) | 22 proto files (services + data types) |
| [rpc/src/api/client.rs](rpc/src/api/client.rs) | RPC client with 20+ methods |
| [rpc/src/api/grpc/](rpc/src/api/grpc/) | gRPC service implementations |
| [rpc/src/utils/rpc_proto_conversions.rs](rpc/src/utils/rpc_proto_conversions.rs) | Proto ↔ domain type conversions |

### Testing

| File | Description |
|------|-------------|
| [test-cluster/src/lib.rs](test-cluster/src/lib.rs) | TestClusterBuilder API |
| [test-cluster/src/swarm.rs](test-cluster/src/swarm.rs) | SwarmBuilder (validators + fullnodes) |
| [e2e-tests/tests/](e2e-tests/tests/) | 14 test files, 93 E2E tests |
| [utils/src/fp.rs](utils/src/fp.rs) | Failpoint infrastructure |

### Configuration Files

| File | Purpose |
|------|---------|
| `Cargo.toml` | Workspace configuration (25 crates) |
| `rustfmt.toml` | Code formatting |
| `flake.nix` | Nix development environment |
| `pyproject.toml` | Python project configuration |
| [REDESIGN.md](REDESIGN.md) | Full redesign specification |

---

## Glossary

| Term | Definition |
|------|------------|
| **Authority** | A validator node participating in consensus |
| **BPS** | Basis points (1 BPS = 0.01%, 10000 = 100%) |
| **Challenge** | A dispute against a mining submission, resolved by validator tally |
| **Checkpoint** | A certified snapshot of executed transactions |
| **Epoch** | A period with fixed validator committees and protocol parameters |
| **Finality** | Transaction is irreversibly committed (in a certified checkpoint) |
| **Hit** | A successful submission within a target's distance threshold |
| **Miner** | A participant who finds data matching targets |
| **Model** | A registered ML model with staked weight, used for embedding computation |
| **msim** | Deterministic simulator for testing (Mysten Sim) |
| **Quorum** | 2f+1 participants (where f is max Byzantine failures) |
| **Safe Mode** | Degraded operation after failed epoch transition (no rewards/targets, but chain continues) |
| **Shannon** | Smallest unit of SOMA (1 SOMA = 1,000,000,000 shannons) |
| **StakedSoma** | User's staked tokens (earns rewards, enables delegation) |
| **Submission** | A miner's data submission to fill a target |
| **Target** | A generated embedding point with radius that miners compete to fill |
| **Tally** | Accumulation of validator reports on a shared object, checked at claim time |
