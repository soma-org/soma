# Soma Monorepo Architecture Guide

This document provides a comprehensive overview of the Soma blockchain and distributed encoding network. It is intended for developers who need to understand the system architecture to orchestrate refactoring or major changes.

## Table of Contents
1. [System Overview](#system-overview)
2. [Crate Organization](#crate-organization)
3. [Core Data Model](#core-data-model)
4. [Validator Flow (Transaction Execution)](#validator-flow-transaction-execution)
5. [Encoder Flow (Embedding Generation)](#encoder-flow-embedding-generation)
6. [User Experience (CLI/SDK/RPC)](#user-experience-clisdkrpc)
7. [Key Architectural Patterns](#key-architectural-patterns)
8. [File Reference](#file-reference)

---

## System Overview

Soma is a **dual-layer consensus system** combining:

1. **Validator Consensus Layer**: Traditional BFT consensus (Mysticeti) for transaction ordering and finality
2. **Encoder Evaluation Layer**: Distributed ML-based evaluation of data embeddings through stake-weighted shard competition

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                 │
│                     CLI / SDK / RPC (gRPC + JSON)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│         FULLNODE LAYER           │  │        ENCODER NETWORK           │
│  ┌────────────────────────────┐  │  │  ┌────────────────────────────┐  │
│  │    Transaction Orchestrator │  │  │  │     Encoder Node           │  │
│  │    (submission + finality)  │  │  │  │  ┌──────────────────────┐  │  │
│  └────────────────────────────┘  │  │  │  │  Commit-Reveal        │  │  │
│               │                  │  │  │  │  Consensus Pipeline   │  │  │
│               ▼                  │  │  │  └──────────────────────┘  │  │
│  ┌────────────────────────────┐  │  │  │  ┌──────────────────────┐  │  │
│  │      Authority State        │  │  │  │  │  Intelligence        │  │  │
│  │  (execution + caching)      │  │  │  │  │  (inference/eval)    │  │  │
│  └────────────────────────────┘  │  │  │  └──────────────────────┘  │  │
│               │                  │  │  │  ┌──────────────────────┐  │  │
│               ▼                  │  │  │  │  Probes (ML models)   │  │  │
│  ┌────────────────────────────┐  │  │  │  └──────────────────────┘  │  │
│  │    Consensus Manager        │  │  │  └────────────────────────┘  │
│  │    (Mysticeti BFT)          │  │  └──────────────────────────────┘
│  └────────────────────────────┘  │
│               │                  │
│               ▼                  │           ▲
│  ┌────────────────────────────┐  │           │
│  │    Checkpoint Service       │◄─┼───────────┘
│  │    (finality + sync)        │  │   encoder-validator-api
│  └────────────────────────────┘  │   (committee sync)
└──────────────────────────────────┘
```

### Data Flow Summary

1. **User submits data** via `soma embed --url <data_url>`
2. **Transaction** enters validator consensus → checkpoint finality
3. **FinalityProof** sent to selected encoder shard (via ShardAuthToken)
4. **Encoders compete** through commit-reveal consensus to produce best embedding
5. **Winner reported** back to chain via ReportWinner transaction
6. **Rewards distributed** to winning encoder and data submitter

---

## Crate Organization

### Core Infrastructure (32 crates total)

| Category | Crate | Purpose |
|----------|-------|---------|
| **Types & Config** | `types` | All shared data structures, transactions, objects, crypto |
| | `protocol-config` | Protocol version configuration |
| | `protocol-config-macros` | Procedural macros for protocol config |
| **Consensus** | `consensus` | Mysticeti BFT consensus implementation |
| | `authority` | Validator state, execution engine, transaction orchestration |
| | `node` | Fullnode orchestration (starts all components) |
| **Encoder System** | `encoder` | Encoder node with commit-reveal pipeline |
| | `intelligence` | ML inference and evaluation services |
| | `probes` | Transformer probe model (Rust/Burn implementation) |
| | `encoder-validator-api` | gRPC API for encoder-validator committee sync |
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
| **Utilities** | `utils` | Shared utilities (compression, logging, codec) |
| | `vdf` | Verifiable Delay Functions (Wesolowski) |
| | `data-ingestion` | Data ingestion pipelines |
| **Python** | `python-probes` | JAX/Flax probe implementation (training) |
| | `arrgen` | Deterministic array generation for testing |
| **Testing** | `test-cluster` | Validator cluster simulation |
| | `test-encoder-cluster` | Encoder cluster simulation |
| | `e2e-tests` | End-to-end integration tests |

### Dependency Graph (Simplified)

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
   └────┬─────┘     └────┬─────┘      └────┬─────┘
        │                │                 │
        └────────────────┼─────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        ┌──────────┐         ┌──────────┐
        │authority │         │ encoder  │
        └────┬─────┘         └────┬─────┘
             │                    │
             ▼                    ▼
        ┌──────────┐         ┌──────────────┐
        │   node   │         │ intelligence │
        └────┬─────┘         └──────────────┘
             │
             ▼
        ┌──────────┐
        │   cli    │
        └──────────┘
```

---

## Core Data Model

### Transaction Types (26 total)

Located in [types/src/transaction.rs](types/src/transaction.rs)

**System Transactions:**
- `Genesis` - Network initialization
- `ConsensusCommitPrologue` - Consensus round marker
- `ChangeEpoch` - Epoch transition

**Validator Management:**
- `AddValidator`, `RemoveValidator`, `UpdateValidatorMetadata`
- `SetCommissionRate`, `ReportValidator`, `UndoReportValidator`

**Encoder Management:**
- `AddEncoder`, `RemoveEncoder`, `UpdateEncoderMetadata`
- `SetEncoderCommissionRate`, `SetEncoderBytePrice`
- `ReportEncoder`, `UndoReportEncoder`

**User Operations:**
- `TransferCoin`, `PayCoins`, `TransferObjects`
- `AddStake`, `AddStakeToEncoder`, `WithdrawStake`

**Shard Operations:**
- `EmbedData` - Submit data for encoding
- `ClaimEscrow` - Claim failed shard escrow
- `ReportWinner` - Report shard winner
- `ClaimReward` - Claim target reward

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
| `SystemState` | Global blockchain state, validators, encoders, staking | Mutable |
| `Shard` | Individual shard with escrow and winner info | Mutable |
| `Target` | Competition target with reward | Mutable |
| `StakedSoma` | User's staked tokens | Mutable |

### Shard Data Flow

```
EmbedData Transaction
        │
        ▼
┌───────────────────┐
│   ShardInput      │ ← Created object (owned by system)
│   - download_meta │
│   - amount        │
│   - submitter     │
│   - target (opt)  │
└───────────────────┘
        │
        │ FinalityProof + VDF
        ▼
┌───────────────────┐
│   ShardAuthToken  │ ← Proves tx finality to encoders
│   - finality_proof│
│   - checkpoint_ent│
│   - vdf_proof     │
│   - shard_ref     │
└───────────────────┘
        │
        │ Encoder competition
        ▼
┌───────────────────┐
│   Shard (final)   │ ← Updated with winner
│   - winning_enc   │
│   - embeddings    │
│   - eval_scores   │
└───────────────────┘
```

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
    │   ├─ Start TransactionOrchestrator
    │   └─ Start EncoderValidatorService
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
│     - ValidatorExecutor, EncoderExecutor  │
│     - CoinExecutor, ShardExecutor, etc.   │
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

Located in [node/src/lib.rs:1047](node/src/lib.rs) - `monitor_reconfiguration()`

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

## Encoder Flow (Embedding Generation)

### Encoder Node Architecture

Located in [encoder/src/core/encoder_node.rs](encoder/src/core/encoder_node.rs)

```
EncoderNode
    │
    ├─ Context (epoch committees, config)
    ├─ Storage (RocksDB for shard state)
    │
    ├─ Network Services:
    │   ├─ External Service (receives inputs from fullnodes)
    │   ├─ Internal Service (peer-to-peer with other encoders)
    │   └─ Object Service (HTTP server for data download)
    │
    ├─ ML Services:
    │   ├─ Inference Network Manager
    │   └─ Evaluation Network Manager
    │
    └─ Pipeline Actors (one per consensus stage):
        ├─ InputProcessor
        ├─ CommitProcessor
        ├─ CommitVotesProcessor
        ├─ RevealProcessor
        ├─ EvaluationProcessor
        └─ ReportVoteProcessor
```

### Commit-Reveal Consensus Pipeline

This is the core encoding workflow. Located in [encoder/src/pipelines/](encoder/src/pipelines/)

```
                        ┌─────────────────────────────────┐
                        │       INPUT STAGE               │
                        │  - Receive ShardAuthToken       │
                        │  - Verify shard membership      │
                        │  - Download input data          │
                        │  - Run inference → embeddings   │
                        │  - Run evaluation → scores      │
                        │  - Create Submission            │
                        │  - Broadcast Commit (hash)      │
                        └───────────────┬─────────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────────┐
                        │       COMMIT STAGE              │
                        │  - Collect commits from peers   │
                        │  - Wait for quorum (2f+1)       │
                        │  - Start timer (min 5 sec)      │
                        └───────────────┬─────────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────────┐
                        │    COMMIT_VOTES STAGE           │
                        │  - Broadcast CommitVotes        │
                        │  - Aggregate acceptance votes   │
                        │  - Determine ACCEPTED/REJECTED  │
                        │    for each encoder             │
                        └───────────────┬─────────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────────┐
                        │       REVEAL STAGE              │
                        │  - Broadcast full Submission    │
                        │  - Verify reveals match commits │
                        │  - Wait for quorum of reveals   │
                        └───────────────┬─────────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────────┐
                        │     EVALUATION STAGE            │
                        │  - Re-evaluate top submissions  │
                        │  - Verify reported scores       │
                        │  - Pick highest-scoring winner  │
                        │  - Create signed Report         │
                        │  - Broadcast ReportVote         │
                        └───────────────┬─────────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────────┐
                        │    REPORT_VOTES STAGE           │
                        │  - Collect ReportVotes          │
                        │  - Aggregate BLS signatures     │
                        │  - Submit ReportWinner tx       │
                        │  - Cleanup shard state          │
                        └─────────────────────────────────┘
```

### Shard Selection

Located in [types/src/encoder_committee.rs](types/src/encoder_committee.rs)

```rust
// Deterministic weighted random sampling
EncoderCommittee::sample_shard(seed: &Digest<ShardEntropy>) -> Shard {
    // 1. Use seed to initialize RNG
    // 2. Sample `shard_size` encoders weighted by stake
    // 3. Return Shard with quorum_threshold = 2f+1
}
```

### Intelligence Services

Located in [intelligence/src/](intelligence/src/)

**Inference Service:**
- Takes input data → produces embeddings
- Supports mock or external JSON/HTTP inference engines
- Returns `InferenceOutput` with embedding download metadata

**Evaluation Service:**
- Takes embeddings + probe model → computes scores
- Uses Burn framework with WGPU backend
- Returns `EvaluationScores`: flow_matching, sig_reg, compression, composite

### Probe Architecture

Located in [probes/src/v1/](probes/src/v1/)

```
Probe (Pre-norm Transformer)
├── Encoder (4 layers)
│   └── Layer (repeated 4x)
│       ├── LayerNorm (norm_1)
│       ├── MultiHeadAttention (8 heads, 1024 dim)
│       │   ├── Query: Linear(1024 → 1024)
│       │   ├── Key:   Linear(1024 → 1024)
│       │   ├── Value: Linear(1024 → 1024)
│       │   └── Output: Linear(1024 → 1024)
│       ├── LayerNorm (norm_2)
│       └── PositionWiseFeedForward
│           ├── Linear(1024 → 4096)
│           ├── GELU activation
│           └── Linear(4096 → 1024)
└── SIGReg (Signature Regularization Loss)
```

**V1 Hyperparameters:**
- `EMBEDDING_DIM = 1024`
- `NUM_HEADS = 8`
- `NUM_LAYERS = 4`
- `MAX_SEQ_LEN = 256`
- `PWFF_HIDDEN_DIM = 4096`

---

## User Experience (CLI/SDK/RPC)

### CLI Commands

Located in [cli/src/](cli/src/)

**User Operations:**
```bash
soma balance [ADDRESS]              # Check balance
soma send --to <ADDR> --amount <N>  # Transfer SOMA
soma transfer --to <ADDR> --object-id <ID>  # Transfer object
soma pay --recipients <A1,A2> --amounts <N1,N2>  # Multi-pay
soma stake --validator <ADDR>       # Stake with validator
soma stake --encoder <ADDR>         # Stake with encoder
soma unstake <STAKED_ID>            # Withdraw stake
soma embed --url <DATA_URL>         # Submit data for encoding
soma claim --escrow <SHARD_ID>      # Claim failed escrow
soma claim --reward <TARGET_ID>     # Claim target reward
```

**Query Commands:**
```bash
soma objects get <ID>               # Get object
soma objects list [OWNER]           # List owned objects
soma tx <DIGEST>                    # Get transaction
soma shards status <ID>             # Shard status
soma shards list --submitter <ADDR> # List shards by submitter
```

**Operator Commands:**
```bash
soma validator start --config <PATH>
soma validator join-committee <INFO_PATH>
soma validator leave-committee
soma encoder start --config <PATH>
soma encoder join-committee <INFO_PATH> --probe-url <URL>
```

### SDK Usage

Located in [sdk/src/](sdk/src/)

```rust
// Initialize client
let client = SomaClientBuilder::default()
    .request_timeout(Duration::from_secs(60))
    .build_testnet()
    .await?;

// Build and execute transaction
let tx_data = builder.build_transaction_data(sender, kind, gas_ref).await?;
let tx = context.sign_transaction(&tx_data).await;
let response = client.execute_transaction(&tx).await?;

// Wait for finality
let (response, completion) = client
    .execute_embed_data_and_wait_for_completion(&tx, timeout)
    .await?;
```

### RPC Services

Located in [rpc/src/](rpc/src/)

**LedgerService (11 methods):**
- `GetServiceInfo`, `GetObject`, `BatchGetObjects`
- `GetTransaction`, `BatchGetTransactions`
- `GetCheckpoint`, `GetEpoch`
- `GetShardsByEpoch`, `GetShardsBySubmitter`, `GetShardsByEncoder`
- `GetClaimableEscrows`, `GetValidTargets`, `GetClaimableRewards`

**StateService (2 methods):**
- `ListOwnedObjects`, `GetBalance`

**TransactionExecutionService (3 methods):**
- `ExecuteTransaction`, `SimulateTransaction`, `InitiateShardWork`

**SubscriptionService (1 method):**
- `SubscribeCheckpoints` (streaming)

---

## Key Architectural Patterns

### 1. Envelope Pattern (Signatures)

Used throughout for authenticated messages:

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
#[enum_dispatch(SubmissionAPI)]
pub enum Submission {
    V1(SubmissionV1),
    // Future: V2(SubmissionV2),
}
```

### 3. Per-Epoch State Isolation

```rust
struct AuthorityState {
    epoch_store: ArcSwap<AuthorityPerEpochStore>,  // Swapped at epoch boundary
    execution_lock: RwLock<EpochId>,               // Read during execution
}
```

### 4. Actor-Based Concurrency

Encoder pipeline uses bounded actor queues:

```rust
struct ActorManager<T: Actor> {
    sender: mpsc::Sender<T::Input>,
    // Buffer size: 100, process one at a time
}
```

### 5. Finality Proofs (Off-Chain Verification)

```rust
struct FinalityProof {
    transaction: Transaction,
    effects: TransactionEffects,
    checkpoint: CertifiedCheckpointSummary,
    inclusion_proof: CheckpointInclusionProof,  // Merkle proof
}
```

### 6. Stake-Weighted Selection

```rust
impl EncoderCommittee {
    fn sample_shard(&self, seed: &Digest) -> Shard {
        // Weighted random sampling by voting_power
        // Deterministic given seed (VDF output)
    }
}
```

---

## File Reference

### Critical Files for Refactoring

| Area | File | Description |
|------|------|-------------|
| **Transactions** | [types/src/transaction.rs](types/src/transaction.rs) | All 26 transaction types |
| **Execution** | [authority/src/execution/mod.rs](authority/src/execution/mod.rs) | Transaction execution pipeline |
| **Type Executors** | [authority/src/execution/shard.rs](authority/src/execution/shard.rs) | Shard-specific execution |
| **System State** | [types/src/system_state/mod.rs](types/src/system_state/mod.rs) | Global state management |
| **Shard Types** | [types/src/shard.rs](types/src/shard.rs) | Shard, ShardAuthToken, Input |
| **Encoder Committee** | [types/src/encoder_committee.rs](types/src/encoder_committee.rs) | Shard sampling, committee |
| **Finality** | [types/src/finality.rs](types/src/finality.rs) | FinalityProof, inclusion proofs |
| **Report/Submission** | [types/src/report.rs](types/src/report.rs), [submission.rs](types/src/submission.rs) | Encoder outputs |
| **Encoder Pipeline** | [encoder/src/pipelines/](encoder/src/pipelines/) | All 6 consensus stages |
| **Intelligence** | [intelligence/src/evaluation/](intelligence/src/evaluation/) | ML evaluation logic |
| **Probes** | [probes/src/v1/](probes/src/v1/) | Transformer model |
| **RPC Proto** | [rpc/src/proto/](rpc/src/proto/) | All protobuf definitions |
| **Node Bootstrap** | [node/src/lib.rs](node/src/lib.rs) | Fullnode/validator startup |

### Testing Entry Points

| Test Type | Location |
|-----------|----------|
| Unit Tests | Each crate's `tests/` directory |
| Integration | [e2e-tests/](e2e-tests/) |
| Cluster Simulation | [test-cluster/](test-cluster/), [test-encoder-cluster/](test-encoder-cluster/) |
| Python Tests | [python-probes/tests/](python-probes/tests/) |

### Configuration Files

| File | Purpose |
|------|---------|
| `Cargo.toml` | Workspace configuration (32 crates) |
| `rustfmt.toml` | Code formatting |
| `flake.nix` | Nix development environment |
| `pyproject.toml` | Python project configuration |

---

## Glossary

| Term | Definition |
|------|------------|
| **Authority** | A validator node participating in consensus |
| **Checkpoint** | A certified snapshot of executed transactions |
| **Encoder** | A node that generates embeddings and participates in shard consensus |
| **Epoch** | A period of time with fixed validator/encoder committees |
| **Finality** | Transaction is irreversibly committed (in a certified checkpoint) |
| **Probe** | A transformer model used to evaluate embedding quality |
| **Quorum** | 2f+1 participants (where f is max Byzantine failures) |
| **Shard** | A subset of encoders selected to process a data submission |
| **ShardAuthToken** | Proof of transaction finality for encoder shard work |
| **StakedSoma** | User's staked tokens (earns rewards, enables delegation) |
| **Target** | A competition goal with associated reward |
| **VDF** | Verifiable Delay Function (provides unpredictable randomness) |

---

## Build Commands

```bash
# Build all crates
cargo build --release

# Run tests
cargo test

# Run specific crate tests
cargo test -p types
cargo test -p encoder

# Build Python probes
cd python-probes && pip install -e .

# Run Python tests
pytest python-probes/tests/

# Start local network
cargo run --bin soma -- start --force-regenesis

# Run E2E tests
cargo test -p e2e-tests
```
