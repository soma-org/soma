# Soma Monorepo Architecture Guide

This document provides a comprehensive overview of the Soma blockchain and its redesign toward a continuous mining competition. It is intended for developers who need to understand the system architecture to orchestrate implementation work.

## Table of Contents
1. [System Overview](#system-overview)
2. [Current State (Post-Cleanup)](#current-state-post-cleanup)
3. [Redesign Overview](#redesign-overview)
4. [Crate Organization](#crate-organization)
5. [Core Data Model (Current)](#core-data-model-current)
6. [Validator Flow (Transaction Execution)](#validator-flow-transaction-execution)
7. [Key Architectural Patterns](#key-architectural-patterns)
8. [Implementation Roadmap](#implementation-roadmap)
9. [File Reference](#file-reference)

---

## System Overview

Soma is a **BFT consensus blockchain** (Mysticeti) currently being redesigned from a dual-layer encoder system into a **continuous mining competition** where data miners race to fill targets using registered models.

### Current Architecture (Post-Cleanup)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                 │
│                     CLI / SDK / RPC (gRPC + JSON)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          FULLNODE LAYER                                   │
│  ┌────────────────────────────┐  ┌────────────────────────────────────┐  │
│  │    Transaction Orchestrator │  │      Authority State               │  │
│  │    (submission + finality)  │  │  (execution + caching)             │  │
│  └────────────────────────────┘  └────────────────────────────────────┘  │
│               │                                    │                     │
│               ▼                                    ▼                     │
│  ┌────────────────────────────┐  ┌────────────────────────────────────┐  │
│  │    Consensus Manager        │  │    Checkpoint Service              │  │
│  │    (Mysticeti BFT)          │  │    (finality + sync)               │  │
│  └────────────────────────────┘  └────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

The old encoder network, shard system, VDF randomness, and encoder-validator API have all been removed. The codebase is a clean validator-only chain ready for the new mining system.

---

## Current State (Post-Cleanup)

The encoder/shard infrastructure has been fully removed across three cleanup commits:

- `02ddf68` - Removed encoder-related crates, upgraded packages, moved to Rust 2024
- `508f3ad` - Removed VDF, enc-val-api crates. Removed shard & encoder things from types and system state
- `d704d9e` - Cleaned authority, rpc, node, sdk, and cli of old encoder and shard related things

### What Was Removed

| Component | Status |
|-----------|--------|
| `encoder/` crate | Deleted |
| `encoder-validator-api/` crate | Deleted |
| `test-encoder-cluster/` crate | Deleted |
| `vdf/` crate | Deleted |
| All shard types (`shard.rs`, `shard_networking/`, `shard_crypto/`, etc.) | Deleted |
| All encoder types (`encoder_committee.rs`, `encoder_info.rs`, `encoder_validator/`, etc.) | Deleted |
| System state encoder/shard fields | Deleted |
| Encoder/shard transaction variants (11 variants removed) | Deleted |
| Authority encoder client and shard/encoder executors | Deleted |
| CLI encoder commands | Deleted |

### What Remains (Post Phase 3)

The chain is a validator blockchain with a model registry system:

- **24 transaction types**: Genesis, ConsensusCommitPrologue, ChangeEpoch, validator management (6), coin/object operations (3), staking (2), model management (9)
- **SystemState** with validators, emission pool, parameters, epoch seeds, model registry
- **Model system**: Commit-reveal lifecycle, staking pools, delegation, epoch-boundary processing
- **Full consensus pipeline**: Mysticeti BFT → checkpoint finality → state sync
- **RPC/CLI/SDK**: Functional for balance, send, stake, validator, and model operations

### Minor Cleanup Leftovers

- `intelligence/`, `probes/`, `arrgen/` crates still in workspace (retained for future use by inference engine)

---

## Redesign Overview

The system is being redesigned from an encoder shard competition into a **continuous mining competition**. Full details in [REDESIGN.md](REDESIGN.md), implementation plan in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

### New System at a Glance

| Aspect | Old (Encoder) | New (Mining + Challenge) |
|--------|---------------|--------------------------|
| **Model Privacy** | Private per-encoder probes | Commit-reveal: commit epoch N, reveal epoch N+1 |
| **Model Selection** | VDF-based random assignment | Stake-weighted random selection |
| **Verification** | Every shard evaluated by committee | Optimistic: only challenged submissions audited |
| **Competition** | Encoder shards race | Miners race to fill targets (first valid commit wins) |
| **Rewards** | Winning encoder + submitter | Miner (50%) + Model owner (30%) + Validators (20%) |

### New Data Flow

1. **Model Registration**: Model owners commit model weights (epoch N) → reveal (epoch N+1) → model becomes active
2. **Target Generation**: Targets generated continuously; each selects one model via stake-weighted sampling with a difficulty-controlled radius
3. **Mining**: Miners find data that embeds within target radius → hash-commit → ~30s reveal window → first valid commit wins
4. **Challenge**: Hits challengeable through epoch N+1; validators audit on challenge; 2f+1 vote resolves
5. **Rewards**: At epoch N+1 end, confirmed hits split epoch emissions (50% miner / 30% model / 20% validators)

---

## Crate Organization

### Current Workspace (25 crates)

| Category | Crate | Purpose |
|----------|-------|---------|
| **Types & Config** | `types` | All shared data structures, transactions, objects, crypto |
| | `protocol-config` | Protocol version configuration |
| | `protocol-config-macros` | Procedural macros for protocol config |
| **Consensus** | `consensus` | Mysticeti BFT consensus implementation |
| | `authority` | Validator state, execution engine, transaction orchestration |
| | `node` | Fullnode orchestration (starts all components) |
| **Storage** | `store` | RocksDB-backed persistent storage |
| | `store-derive` | Derive macros for storage traits |
| | `objects` | Object storage abstraction (S3, GCS, Azure, HTTP) |
| **Networking** | `sync` | P2P state synchronization |
| | `soma-http` | HTTP client/server layer |
| | `soma-tls` | TLS certificate management |
| | `soma-keys` | Cryptographic key management |
| | `rpc` | gRPC/JSON-RPC service definitions |
| **User Interface** | `cli` | Command-line interface (`soma` binary) |
| | `sdk` | Rust client SDK |
| | `python-sdk` | Python bindings (PyO3) |
| **ML (retained)** | `intelligence` | ML inference and evaluation services |
| | `probes` | Transformer probe model (Rust/Burn implementation) |
| **Utilities** | `utils` | Shared utilities (compression, logging, codec) |
| | `data-ingestion` | Data ingestion pipelines |
| | `arrgen` | Deterministic array generation for testing |
| | `bin-version` | Binary version information |
| **Testing** | `test-cluster` | Validator cluster simulation |
| | `e2e-tests` | End-to-end integration tests |

### Dependency Graph

```
                    ┌──────────────┐
                    │    types     │ ◄──────── All crates depend on types
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐      ┌──────────┐
   │  store   │     │  crypto  │      │   rpc    │
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

## Core Data Model (Current)

### Transaction Types (24 total)

Located in [types/src/transaction.rs](types/src/transaction.rs)

**System Transactions:**
- `Genesis` - Network initialization
- `ConsensusCommitPrologue` - Consensus round marker
- `ChangeEpoch` - Epoch transition

**Validator Management:**
- `AddValidator`, `RemoveValidator`, `UpdateValidatorMetadata`
- `SetCommissionRate`, `ReportValidator`, `UndoReportValidator`

**Coin/Object Operations:**
- `TransferCoin`, `PayCoins`, `TransferObjects`

**Staking:**
- `AddStake`, `WithdrawStake`

**Model Management (9):**
- `CommitModel` - Register a new model (commit phase)
- `RevealModel` - Reveal model weights (reveal phase)
- `CommitModelUpdate` - Commit updated weights for an active model
- `RevealModelUpdate` - Reveal updated weights
- `AddStakeToModel` - Stake tokens to a model's staking pool
- `SetModelCommissionRate` - Set model's commission rate for next epoch
- `DeactivateModel` - Voluntarily deactivate a model
- `ReportModel` - Active validator reports a model
- `UndoReportModel` - Remove a model report

### Object Model

```
Object
├── ObjectID (32 bytes)
├── Version (u64, lamport timestamp)
├── Digest (hash of contents)
├── Owner
│   ├── AddressOwner(SomaAddress)    # User-owned
│   ├── ObjectOwner(ObjectID)         # Nested ownership
│   ├── Shared { initial_version }    # Consensus-sequenced
│   └── Immutable                     # Read-only
└── Data (BCS-serialized contents)
```

### Key Shared Objects

| Object | Purpose | Mutability |
|--------|---------|------------|
| `SystemState` | Global blockchain state, validators, staking, emissions, model registry | Mutable |
| `StakedSoma` | User's staked tokens (for validators or models) | Mutable |

---

## Validator Flow (Transaction Execution)

### Node Startup (`node/src/lib.rs`)

```
SomaNode::start(config)
    │
    ├─ Load genesis + protocol config
    ├─ Open RocksDB stores (perpetual, checkpoint, committee)
    ├─ Create AuthorityState
    ├─ Setup P2P network (Discovery + StateSync)
    │
    ├─ IF VALIDATOR:
    │   ├─ Start ConsensusManager (Mysticeti)
    │   ├─ Start CheckpointService
    │   └─ Start gRPC validator service
    │
    ├─ IF FULLNODE:
    │   ├─ Start RPC servers (gRPC + JSON)
    │   └─ Start TransactionOrchestrator
    │
    └─ Spawn reconfiguration monitor (runs forever)
```

### Transaction Execution Pipeline

Located in [authority/src/execution/mod.rs](authority/src/execution/mod.rs)

```
Transaction Received
        │
        ▼
┌───────────────────────────────────────────┐
│  1. Gas Preparation                       │
│     - Validate gas payment                │
│     - Deduct base transaction fee         │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  2. Input Loading                         │
│     - Load owned objects (by ObjectRef)   │
│     - Load shared objects (with versions) │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  3. Execute Transaction                   │
│     - Dispatch to type-specific executor  │
│     - ValidatorExecutor, CoinExecutor     │
│     - StakingExecutor, ModelExecutor      │
│     - SystemExecutor                      │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  4. Generate Effects                      │
│     - Changed objects (created/mutated)   │
│     - Deleted objects                     │
│     - Gas cost breakdown                  │
│     - Execution status                    │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  5. Commit to Storage                     │
│     - Write to epoch store                │
│     - Write to perpetual store            │
│     - Update caches                       │
└───────────────────────────────────────────┘
```

### Consensus Integration

Located in [authority/src/consensus_handler.rs](authority/src/consensus_handler.rs)

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
ExecutionScheduler
        │
        ├─ Queue transactions for execution
        ├─ Respect dependency ordering
        │
        ▼
ExecutionDriver
        │
        └─ Execute via AuthorityState::try_execute_immediately()
```

### Epoch Reconfiguration

Located in `node/src/lib.rs` - `monitor_reconfiguration()`

```
End of Epoch Detected
        │
        ├─ CheckpointExecutor runs all remaining checkpoints
        ├─ Compute global state hash
        │
        ▼
Reconfigure State
        │
        ├─ Create new AuthorityPerEpochStore
        ├─ Load next epoch committee
        │
        ├─ IF STILL VALIDATOR:
        │   ├─ Shutdown old consensus
        │   ├─ Start new consensus
        │   └─ Restart checkpoint service
        │
        └─ Continue to next epoch
```

---

## Key Architectural Patterns

### 1. Envelope Pattern (Signatures)

```rust
struct Envelope<T: Message, S: Signature> {
    data: T,
    auth_signature: S,
}

type CertifiedTransaction = Envelope<SenderSignedData, AuthorityStrongQuorumSignInfo>;
type CertifiedCheckpoint = Envelope<CheckpointSummary, AuthorityStrongQuorumSignInfo>;
```

### 2. Versioned Types (Protocol Evolution)

```rust
#[enum_dispatch(SomeAPI)]
pub enum SomeType {
    V1(SomeTypeV1),
    // Future: V2(SomeTypeV2),
}
```

### 3. Per-Epoch State Isolation

```rust
struct AuthorityState {
    epoch_store: ArcSwap<AuthorityPerEpochStore>,  // Swapped at epoch boundary
    execution_lock: RwLock<EpochId>,               // Read during execution
}
```

### 4. FullnodeConfigBuilder (Test Infrastructure)

Fullnodes in test clusters are created independently from validators using `FullnodeConfigBuilder` in `types/src/config/node_config.rs`. This replaces the previous `CommitteeConfig::Mixed` approach which incorrectly added fullnodes to the validator committee.

```rust
let fullnode_config = FullnodeConfigBuilder::new()
    .with_config_directory(dir)
    .build(genesis, seed_peers);
// fullnode_config.consensus_config == None → node runs as fullnode
```

Key details:
- Generates independent keypairs (not in validator committee)
- Sets `consensus_config = None` so the node runs as a fullnode
- In msim, uses `local_ip_utils::get_new_ip()` for unique simulated IPs (`10.10.0.X`)
- Connects to validators via `seed_peers` extracted from validator configs
- Used by `SwarmBuilder::build()` in `test-cluster/src/swarm.rs`

### 5. Finality Proofs (Off-Chain Verification)

```rust
struct FinalityProof {
    transaction: Transaction,
    effects: TransactionEffects,
    checkpoint: CertifiedCheckpointSummary,
    inclusion_proof: CheckpointInclusionProof,  // Merkle proof
}
```

---

## Implementation Roadmap

Phases 1 (cleanup) and 3 (model registry) are complete. The remaining phases from [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md):

### Phase 2: Step-Decay Emissions
Update `EmissionPool` with step-decay schedule (subsidy period + decrease rate). Split emissions 20% validators / 80% target winner pool.

### Phase 3: Model Registry with Staking — COMPLETE ✓
`Model` type with commit-reveal flow. `ModelRegistry` in SystemState. 9 model transactions with `ModelExecutor`. Genesis bootstrap with seed models. Full CLI (`soma model commit|reveal|list|info|stake`). RPC proto integration. 18 unit tests + 2 E2E tests passing.

### Phase 4: Target Generation
New `Target` type with deterministic embedding generation and stake-weighted model selection. `TargetState` and `DifficultyStats` in SystemState. Continuous generation: new target spawns when previous is filled. Genesis creates initial targets.

### Phase 5: Submission System
New `Submission` type with hash-commit-reveal. First valid commit wins. ~30s reveal window. Lazy cleanup of expired commits. `HitTracking` in SystemState. New transactions: `CommitSubmission`, `RevealSubmission`, `ClaimBond`. CLI: `soma submit commit|reveal|claim-bond`.

### Phase 6: Inference Engine
New `inference-engine/` crate wrapping `probes/`. Shared by miners (CLI), challengers, and validators. Deterministic f32 inference with fixed-point comparisons.

### Phase 7: Challenge System
New `Challenge` type with `ScoreFraud` and `DataFraud` types. Validator audit service downloads data, re-runs inference, votes. 2f+1 quorum resolves. New transactions: `InitiateChallenge`, `ResolveChallengeAudit`. CLI: `soma challenge initiate|list|info`.

### Phase 8: Reward Distribution & Difficulty Adjustment
At epoch N+1 end: close challenge window for epoch N, distribute rewards for confirmed hits, adjust difficulty radius based on average hit time.

### Phase 9: Error Handling
New `ExecutionFailureStatus` variants for model, target, submission, challenge, and bond errors.

### Phase 10: RPC & SDK Updates
Index tables for models, targets, submissions, challenges. New proto query methods. SDK client methods for full flow.

### Phase 11: Polish & Testnet
Comprehensive E2E tests, documentation, testnet deployment with seed models.

---

## File Reference

### Current Critical Files

| Area | File | Description |
|------|------|-------------|
| **Transactions** | [types/src/transaction.rs](types/src/transaction.rs) | 24 transaction types (system, validator, coin, staking, model) |
| **Execution** | [authority/src/execution/mod.rs](authority/src/execution/mod.rs) | Transaction execution pipeline + executor dispatch |
| **Executors** | [authority/src/execution/validator.rs](authority/src/execution/validator.rs) | Validator management |
| | [authority/src/execution/coin.rs](authority/src/execution/coin.rs) | Coin operations |
| | [authority/src/execution/staking.rs](authority/src/execution/staking.rs) | Staking operations |
| | [authority/src/execution/model.rs](authority/src/execution/model.rs) | Model management (9 transaction types) |
| | [authority/src/execution/change_epoch.rs](authority/src/execution/change_epoch.rs) | Epoch transitions (incl. model epoch processing) |
| **Model Types** | [types/src/model.rs](types/src/model.rs) | Model, ModelId, ModelWeightsDownload, PendingModelUpdate |
| | [types/src/system_state/model_registry.rs](types/src/system_state/model_registry.rs) | ModelRegistry (active, pending, inactive models) |
| **System State** | [types/src/system_state/mod.rs](types/src/system_state/mod.rs) | Global state (validators, emissions, parameters, model registry) |
| **Config** | [types/src/config/node_config.rs](types/src/config/node_config.rs) | ValidatorConfigBuilder + FullnodeConfigBuilder |
| | [types/src/config/network_config.rs](types/src/config/network_config.rs) | Network/committee configuration |
| | [types/src/config/genesis_config.rs](types/src/config/genesis_config.rs) | Genesis config (validators, models, token allocations) |
| **Genesis** | [types/src/genesis_builder.rs](types/src/genesis_builder.rs) | Genesis state creation (validators + seed models) |
| **Finality** | [types/src/finality.rs](types/src/finality.rs) | FinalityProof, inclusion proofs (generic) |
| **Node Bootstrap** | [node/src/lib.rs](node/src/lib.rs) | Fullnode/validator startup |
| **Consensus** | [authority/src/consensus_handler.rs](authority/src/consensus_handler.rs) | Consensus commit handling |
| **Testing** | [test-cluster/src/lib.rs](test-cluster/src/lib.rs) | TestCluster for E2E tests |
| | [test-cluster/src/swarm.rs](test-cluster/src/swarm.rs) | SwarmBuilder (validators + fullnodes via FullnodeConfigBuilder) |
| **RPC Proto** | [rpc/src/proto/](rpc/src/proto/) | Protobuf definitions (incl. model messages) |
| **CLI** | [cli/src/commands/model.rs](cli/src/commands/model.rs) | Model CLI commands |

### Files to Create (Next Phases)

| Crate | File | Phase | Contents |
|-------|------|-------|----------|
| types | `target.rs` | 4 | Target, TargetStatus, TargetState |
| types | `submission.rs` | 5 | Submission (hash-commit-reveal) |
| types | `challenge.rs` | 7 | Challenge, ChallengeType, ChallengeVerdict |
| authority | `execution/target.rs` | 4 | Target generation |
| authority | `execution/submission.rs` | 5 | Submission transaction executor |
| authority | `execution/challenge.rs` | 7 | Challenge transaction executor |
| authority | `challenge_audit.rs` | 7 | Validator audit service |
| new crate | `inference-engine/` | 6 | Shared inference engine wrapping probes |
| cli | `commands/submit.rs` | 5 | Submission CLI commands |
| cli | `commands/challenge.rs` | 7 | Challenge CLI commands |

### Testing Entry Points

| Test Type | Location |
|-----------|----------|
| Unit Tests | Each crate's `tests/` directory |
| Integration | [e2e-tests/](e2e-tests/) |
| Cluster Simulation | [test-cluster/](test-cluster/) |

### Configuration Files

| File | Purpose |
|------|---------|
| `Cargo.toml` | Workspace configuration (25 crates) |
| `rustfmt.toml` | Code formatting |
| `flake.nix` | Nix development environment |
| `pyproject.toml` | Python project configuration |
| `REDESIGN.md` | Full redesign specification |
| `IMPLEMENTATION_PLAN.md` | Phased implementation plan (Phases 1, 3 complete) |

---

## CLI Commands (Current)

Located in [cli/src/](cli/src/)

**User Operations:**
```bash
soma balance [ADDRESS]              # Check balance
soma send --to <ADDR> --amount <N>  # Transfer SOMA
soma transfer --to <ADDR> --object-id <ID>  # Transfer object
soma pay --recipients <A1,A2> --amounts <N1,N2>  # Multi-pay
soma stake --validator <ADDR>       # Stake with validator
soma stake --model <MODEL_ID>       # Stake with model
soma unstake <STAKED_ID>            # Withdraw stake
```

**Model Operations:**
```bash
soma model commit --model-id <ID> --weights-url-commitment <HASH> --weights-commitment <HASH> --architecture-version <N> --stake-amount <N>
soma model reveal --model-id <ID> --weights-url <URL> --weights-checksum <HASH> --weights-size <N> --decryption-key <KEY>
soma model update-commit --model-id <ID> --weights-url-commitment <HASH> --weights-commitment <HASH>
soma model update-reveal --model-id <ID> --weights-url <URL> --weights-checksum <HASH> --weights-size <N> --decryption-key <KEY>
soma model deactivate --model-id <ID>
soma model set-commission-rate --model-id <ID> --rate <BPS>
soma model info <ID>                # Query model details
soma model list                     # List all models
```

**Query Commands:**
```bash
soma objects get <ID>               # Get object
soma objects list [OWNER]           # List owned objects
soma tx <DIGEST>                    # Get transaction
```

**Operator Commands:**
```bash
soma validator start --config <PATH>
soma validator join-committee <INFO_PATH>
soma validator leave-committee
soma validator report-model <MODEL_ID>  # Report a model (validators only)
```

**Genesis Ceremony:**
```bash
soma genesis-ceremony add-model <YAML>  # Add seed model to genesis
soma genesis-ceremony list-models       # List seed models
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Authority** | A validator node participating in consensus |
| **Challenge** | (New) A dispute against a mining submission, resolved by validator audit |
| **Checkpoint** | A certified snapshot of executed transactions |
| **Epoch** | A period of time with fixed validator committees |
| **Finality** | Transaction is irreversibly committed (in a certified checkpoint) |
| **Hit** | (New) A confirmed valid submission within a target's radius |
| **Miner** | (New) A participant who finds data matching targets |
| **Model** | (New) A registered ML model with staked weight, used for embedding computation |
| **Probe** | A transformer model used to evaluate embedding quality (implementation in `probes/`) |
| **Quorum** | 2f+1 participants (where f is max Byzantine failures) |
| **StakedSoma** | User's staked tokens (earns rewards, enables delegation) |
| **Submission** | (New) A miner's hash-committed claim to have filled a target |
| **Target** | (New) A generated embedding point with radius that miners compete to fill |

---

## Build Commands

```bash
# Build all crates
cargo build --release

# Run unit tests
cargo test
cargo test -p types
cargo test -p authority

# Run E2E tests (require msim simulator)
RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test model_tests

# Build with msim (for test-cluster, e2e-tests)
RUSTFLAGS="--cfg msim" cargo build -p e2e-tests -p test-cluster

# Start local network
cargo run --bin soma -- start --force-regenesis
```
