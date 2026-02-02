# SOMA Redesign: Implementation Plan

**Key Context:** Chain is not live. Delete freely, no migration needed.

---

## Overview

### What's Changing

| Aspect | Current (Encoder) | Target (Data Miner + Challenge) |
|--------|-------------------|--------------------------------|
| **Model Privacy** | Private per-encoder probes | Commit-reveal: commit in epoch N, reveal in epoch N+1 |
| **Model Selection** | VDF-based random assignment | Stake-weighted random selection |
| **Verification** | Every shard evaluated by committee | Optimistic: only challenged submissions audited |
| **Competition** | Encoder shards race | Miners race to fill targets (first valid commit wins) |
| **Rewards** | Winning encoder + submitter | Miner (50%) + Model owner (30%) + Validators (20%) |

### Continuous Mining Model

Unlike the previous 3-epoch wave pipeline, the new system operates continuously:

- **Targets**: Generated on-demand when previous targets are filled
- **Submissions**: Hash-commit-reveal with ~30 second reveal window
- **Challenges**: Extended window (epoch N hits challengeable through epoch N+1)
- **Models**: Simple commit-reveal (commit in N, reveal in N+1)

### Data Flow

1. **Model Commit-Reveal:**
   - **Epoch N**: Model owner commits hash of encrypted weights URL + weight hash + stake
   - **Epoch N+1**: Model owner reveals encrypted weights URL + decryption key
   - If not revealed by end of epoch N+1: stake slashed, model removed
   - Model becomes active and eligible for target selection after reveal

2. **Target Generation (continuous):**
   - When a target is filled, new target generated using checkpoint digest as seed
   - Single model selected via stake-weighted random sampling
   - Pre-allocated reward pool from epoch emissions

3. **Mining & Submission (continuous):**
   - Miners find data that embeds within target radius using the assigned model
   - Submit hash commitment + bond
   - First valid commit wins provisionally
   - ~30 second reveal window to prove validity

4. **Challenge (epoch N through N+1):**
   - Anyone can challenge with bond (1/3 of miner bond)
   - Challenge types: ScoreFraud (lied about distance), DataFraud (data mismatch)
   - Validators audit, vote on outcome
   - Successful challenge → miner slashed, target reopens
   - At epoch N+1 end: confirmed hits receive rewards

---

## Phase 1: Core Removals

**Goal:** Codebase compiles without encoder infrastructure.

### 1.1 Remove Crates from Workspace

Edit `Cargo.toml` workspace members, remove:
- `encoder/`, `encoder-validator-api/`, `test-encoder-cluster/`, `intelligence/`
- `vdf/`, `arrgen/`

Note: Keep `soma-http/`, `soma-tls/`, `objects/` as they may be used by RPC layer and object storage for checkpoints.

### 1.2 Delete Transaction Types

In `types/src/transaction.rs`, remove from `TransactionKind`:
```rust
AddEncoder, RemoveEncoder, ReportEncoder, UndoReportEncoder,
UpdateEncoderMetadata, SetEncoderCommissionRate, SetEncoderBytePrice,
AddStakeToEncoder, EmbedData, ClaimEscrow, ReportWinner
```

Also delete: `AddEncoderArgs`, `RemoveEncoderArgs`, `UpdateEncoderMetadataArgs`

### 1.3 Clean SystemState

In `types/src/system_state/mod.rs`, remove:
- `encoders: EncoderSet` field
- `encoder_report_records` field
- `reference_byte_price` field
- All encoder-related methods (`request_add_encoder`, `report_encoder`, etc.)
- `get_current_epoch_encoder_committee()` from trait and impl

Delete files:
- `types/src/system_state/encoder.rs`
- `types/src/system_state/shard.rs`

### 1.4 Delete Type Files

In `types/src/`:
- `encoder_committee.rs`, `encoder_info.rs`, `encoder_validator/`
- `shard.rs`, `report.rs`, `submission.rs` (current versions)
- `shard_networking/`, `shard_crypto/`, `shard_verifier.rs`
- `entropy.rs`, `evaluation.rs`
- `config/encoder_config.rs`

Note: `finality.rs` is already generic (proves checkpoint inclusion for any transaction) - no changes needed.

### 1.5 Clean Authority Crate

Delete:
- `authority/src/execution/encoder.rs`
- `authority/src/execution/shard.rs`
- `authority/src/encoder_client.rs`

Update `authority/src/execution/mod.rs`: remove encoder/shard executor dispatch

### 1.6 Clean Node

In `node/src/lib.rs`, remove:
- `EncoderClientService` initialization
- `EncoderValidatorService` setup
- Encoder committee updates in `monitor_reconfiguration()`

### 1.7 Clean CLI

- Delete `cli/src/commands/encoder.rs`
- Update `cli/src/commands/mod.rs`
- Remove `soma encoder *`, `soma embed`, `soma shards *` commands

### 1.8 Delete Test Files

- `types/src/system_state/unit_tests/encoder_staking.rs`
- `e2e-tests/tests/encoder_committee_tests.rs`

### 1.9 Update Protos

- `rpc/src/proto/soma.proto`: remove encoder-related messages/services
- `rpc/src/proto/authority.proto`: remove encoder-related messages

**Gate:** `cargo build --workspace && cargo test -p types -p authority -p node`

---

## Phase 2: Step-Decay Emissions

**Goal:** Replace linear emissions with logarithmic decay.

### 2.1 Update EmissionPool

In `types/src/system_state/emission.rs`, add fields:
```rust
pub struct EmissionPool {
    pub balance: u64,
    pub distribution_counter: u64,
    pub current_distribution_amount: u64,
    pub subsidy_period_length: u64,      // e.g., 365 epochs
    pub subsidy_decrease_rate_bps: u16,  // e.g., 1000 = 10%
}
```

Update `advance_epoch()`: decay `current_distribution_amount` by `decrease_rate_bps` every `period_length` epochs.

### 2.2 Update Emission Allocation

Split each epoch's emissions:
- 20% to validators (proportional to stake, same as current logic)
- 80% to target winner pool (distributed per-hit as 50% miner / 30% model)

### 2.3 Add Protocol Config Parameters

In `protocol-config/src/lib.rs`:
```rust
initial_emission_per_epoch: Option<u64>,
emission_period_length: Option<u64>,
emission_decrease_rate_bps: Option<u16>,
validator_reward_share_bps: Option<u64>,  // 2000 = 20%
miner_reward_share_bps: Option<u64>,      // 5000 = 50%
model_reward_share_bps: Option<u64>,      // 3000 = 30%
```

### 2.4 Update Genesis Config

In `types/src/config/genesis_config.rs`: set initial emission parameters.

**Gate:** Unit tests for step-decay schedule, allocation split

---

## Phase 3: Model Registry with Staking

**Goal:** Anyone can register models with staking, using simple commit-reveal flow.

### 3.1 Create Model Type

New file `types/src/model.rs`:
```rust
pub struct Model {
    pub id: ModelId,
    pub owner: SomaAddress,
    pub architecture_version: u64,

    // Commit state
    pub weights_url_commitment: Digest<String>,   // hash(encrypted_weights_url)
    pub weight_commitment: Digest<ModelWeights>,  // hash(decrypted weights)
    pub commit_epoch: EpochId,

    // Revealed state (None if pending)
    pub encrypted_weights_url: Option<String>,
    pub decryption_key: Option<[u8; 32]>,

    // Staking
    pub staking_pool: StakingPool,

    // Pending update (if any)
    pub pending_update: Option<PendingModelUpdate>,

    // Audit state (resets each epoch)
    pub tally_count: u64,

    // Lifetime earnings for reward tracking
    pub total_rewards_earned: u64,
}

pub struct PendingModelUpdate {
    pub weights_url_commitment: Digest<String>,
    pub weight_commitment: Digest<ModelWeights>,
    pub commit_epoch: EpochId,
}

pub type ModelId = ObjectID;
```

Note: Model is pending if `encrypted_weights_url.is_none()`, active otherwise.

### 3.2 Create ModelRegistry in SystemState

Add to `types/src/system_state/mod.rs`:
```rust
pub struct ModelRegistry {
    /// All model IDs that have been revealed
    pub active_model_ids: BTreeSet<ModelId>,

    /// Models pending reveal (committed, awaiting reveal)
    pub pending_model_ids: BTreeSet<ModelId>,

    /// Maps staking pool IDs to model IDs
    pub staking_pool_mappings: BTreeMap<ObjectID, ModelId>,

    /// Total stake across all active models (for weighted selection)
    pub total_model_stake: u64,
}
```

### 3.3 Add Model Transactions

In `types/src/transaction.rs`:
```rust
/// Commit a new model (epoch N)
CommitModel {
    weights_url_commitment: Vec<u8>,      // hash(encrypted_weights_url)
    weight_commitment: Vec<u8>,           // hash(decrypted weights)
    architecture_version: u64,
    stake_coin: ObjectRef,                // Initial stake
},

/// Reveal model weights (epoch N+1)
RevealModel {
    model_id: ObjectID,
    encrypted_weights_url: String,
    decryption_key: [u8; 32],
},

/// Add stake to a model
AddStakeToModel {
    model_address: SomaAddress,           // Model owner address
    coin_ref: ObjectRef,
    amount: Option<u64>,
},

/// Commit model weight update
CommitModelUpdate {
    model_id: ObjectID,
    weights_url_commitment: Vec<u8>,
    weight_commitment: Vec<u8>,
},

/// Reveal model weight update
RevealModelUpdate {
    model_id: ObjectID,
    encrypted_weights_url: String,
    decryption_key: [u8; 32],
},

/// Report a model as unavailable (triggers validator audit)
ReportModelUnavailable {
    model_id: ObjectID,
    challenger_bond_coin: ObjectRef,
},
```

### 3.4 Implement Model Executor

New file `authority/src/execution/model.rs`:

**CommitModel:**
- Validate stake >= `model_min_stake`
- Validate architecture_version == current `model_architecture_version`
- Create Model object with reveal fields = None
- Initialize staking pool with provided stake
- Add to `pending_model_ids`

**RevealModel:**
- Validate model exists in `pending_model_ids`
- Validate `current_epoch <= commit_epoch + 1`
- Validate `hash(encrypted_weights_url) == weights_url_commitment`
- Set reveal fields on model
- Move from `pending_model_ids` to `active_model_ids`
- Update `total_model_stake`

**Model Reveal Timeout (in ChangeEpoch):**
- For each model in `pending_model_ids` where `commit_epoch < current_epoch`:
  - Slash stake by `model_reveal_slash_rate_bps`
  - Return slashed amount to emission pool
  - Move staking pool to inactive
  - Remove from `pending_model_ids`

**Model Tally Processing (in ChangeEpoch):**
- For each active model where `tally_count >= model_tally_threshold`:
  - Slash stake by `model_tally_slash_rate_bps`
  - Move to inactive
  - Remove from `active_model_ids`
- Reset `tally_count` to 0 for all remaining models

### 3.5 Add Protocol Config

```rust
model_min_stake: Option<u64>,
model_architecture_version: Option<u64>,
model_max_weight_size: Option<u64>,
model_reveal_slash_rate_bps: Option<u64>,   // e.g., 5000 = 50%
model_tally_threshold: Option<u64>,          // e.g., 3 tallies
model_tally_slash_rate_bps: Option<u64>,     // e.g., 9500 = 95%
embedding_dim: Option<u64>,
```

### 3.6 Add CLI Commands

New file `cli/src/commands/model.rs`:
- `soma model commit --weights <FILE> --stake <AMOUNT>`
  - Encrypts weights, computes commitments, submits tx
- `soma model reveal --model <ID> --weights <FILE>`
  - Reveals encrypted URL and decryption key
- `soma model list [--owner <ADDR>] [--pending]`
- `soma model info <ID>`
- `soma model download --model <ID> --output <DIR>`
  - Downloads encrypted weights, decrypts using on-chain key
- `soma model stake --model <ID> --amount <AMOUNT>`
- `soma model update-commit --model <ID> --weights <FILE>`
- `soma model update-reveal --model <ID> --weights <FILE>`

### 3.7 Genesis Bootstrap Requirements

Genesis must include at least one seed model for target generation to work:

```rust
// In genesis config:
pub struct GenesisModelConfig {
    pub owner: SomaAddress,
    pub encrypted_weights_url: String,     // Pre-published weights
    pub decryption_key: [u8; 32],          // Decryption key (public at genesis)
    pub weight_commitment: Digest<ModelWeights>,
    pub initial_stake: u64,
    pub architecture_version: u64,
}
```

**Genesis Model Handling:**
- Genesis models are created directly in active state (skip commit-reveal)
- `encrypted_weights_url` and `decryption_key` set directly
- Required: At least 1 model in genesis for target generation

**Gate:** E2E test for model commit → reveal → active lifecycle

---

## Phase 4: Target Generation

**Goal:** Continuous target generation with stake-weighted model selection.

### 4.1 Create Target Type

New file `types/src/target.rs`:
```rust
pub struct Target {
    pub id: ObjectID,
    pub embedding: Vec<f32>,
    pub model_id: ModelId,             // Single model selected via stake-weighted sampling
    pub radius: f32,                   // Current difficulty radius
    pub generation_epoch: EpochId,
    pub generation_timestamp_ms: u64,  // For difficulty adjustment calculation

    // State
    pub status: TargetStatus,
    pub winning_submission_id: Option<ObjectID>,
}

pub enum TargetStatus {
    Open,
    Filled { fill_epoch: EpochId },
    Challenged,
}
```

Note: Targets no longer have individual `reward_pool` fields. Rewards are calculated at epoch N+1 boundary as `epoch_emissions * target_share_bps / num_confirmed_hits`.
```

### 4.2 Add Target State to SystemState

In `types/src/system_state/mod.rs`:
```rust
pub struct TargetState {
    /// Currently open targets
    pub open_target_ids: BTreeSet<ObjectID>,

    /// Filled targets awaiting challenge window close
    pub filled_target_ids: BTreeSet<ObjectID>,

    /// Current difficulty radius
    pub current_radius: f32,

    /// Target index counter for seed generation
    pub target_index: u64,
}

/// Tracks difficulty adjustment statistics per epoch
pub struct DifficultyStats {
    /// Sum of hit times (reveal_timestamp - generation_timestamp) for confirmed hits
    pub total_hit_time_ms: u64,
    /// Number of confirmed hits this epoch (for computing average)
    pub confirmed_hit_count: u64,
}
```

Note: `DifficultyStats` is updated during reward distribution at epoch N+1 end, then used to adjust radius.

### 4.3 Implement Target Generation

In `authority/src/execution/target.rs`:

**generate_target()** (called when previous target filled):
```rust
// 1. Compute seed = checkpoint_contents_digest || target_index || nonce
// 2. Generate embedding = deterministic_gaussian(seed, embedding_dim)
// 3. Select model via stake-weighted random sampling from active models
// 4. Create Target with status = Open, generation_timestamp_ms = current_timestamp
// 5. Increment target_index
```

**stake_weighted_model_selection():**
```rust
// Similar to encoder shard selection, but for models:
// 1. Use seed to initialize RNG
// 2. Sample 1 model weighted by staking_pool.soma_balance
// 3. Return selected model_id
```

Note: No reward pool allocation per target. Rewards are computed at epoch end from total emissions.

### 4.4 Add Protocol Config

```rust
target_generation_budget_per_epoch: Option<u64>,
initial_radius: Option<f32>,
min_radius: Option<f32>,
max_radius: Option<f32>,
target_hit_time_ms: Option<u64>,
radius_adjustment_rate: Option<f32>,
```

### 4.5 Initial Target Generation at Genesis

Genesis must create initial targets for mining to begin:

```rust
// In genesis transaction execution:
// 1. Verify at least 1 active model exists
// 2. Generate initial_target_count targets using genesis block digest as seed
// 3. Each target gets generation_timestamp_ms = genesis_timestamp
```

**Genesis Config:**
```rust
pub struct GenesisTargetConfig {
    pub initial_target_count: u64,  // Number of targets to create at genesis
}
```

**Gate:** E2E test for target generation, stake-weighted model selection

---

## Phase 5: Submission System

**Goal:** Hash-commit-reveal submission flow with first-wins semantics.

### 5.1 Create Submission Type

New file `types/src/submission.rs`:
```rust
pub struct Submission {
    pub id: ObjectID,
    pub target_id: ObjectID,
    pub miner: SomaAddress,

    // Committed state
    pub commitment: Digest<SubmissionReveal>,  // hash(data_hash || embedding || nonce)
    pub commit_timestamp_ms: u64,              // From consensus commit prologue
    pub bond_amount: u64,

    // Revealed state (None until reveal)
    pub data_hash: Option<Digest<Vec<u8>>>,
    pub data_url: Option<String>,
    pub embedding: Option<Vec<f32>>,           // Provided by submitter, verified by validators on challenge
    pub distance: Option<f32>,                 // Computed from embedding during reveal
    pub reveal_epoch: Option<EpochId>,
    pub reveal_timestamp_ms: Option<u64>,      // For difficulty adjustment
}
```

Note: `embedding` is provided by the submitter in the reveal. Transaction executor only verifies the L2 distance computation. Validators re-compute the embedding only during challenge audit. `model_id` is not stored since it's always the target's assigned model.

### 5.2 Add Hit Tracking to SystemState

```rust
pub struct HitTracking {
    /// Hits by epoch: epoch -> submission_ids
    pub hits: BTreeMap<EpochId, BTreeSet<ObjectID>>,

    /// Slashed submissions (persists for bond claim prevention)
    pub slashed_submissions: BTreeSet<ObjectID>,

    /// Active challenges: submission_id -> challenge_id
    /// Used to check "no active challenge exists" and for timeout cleanup
    pub pending_challenges: BTreeMap<ObjectID, ObjectID>,
}
```

Note: `HitTracking` is part of `SystemState`. `pending_challenges` is cleaned up during ChangeEpoch when challenge windows close.

### 5.3 Add Submission Transactions

In `types/src/transaction.rs`:
```rust
/// Commit a submission (hash only)
CommitSubmission {
    target_id: ObjectID,
    commitment: Digest<SubmissionReveal>,  // hash(data_hash || embedding || nonce)
    bond_coin: ObjectRef,
},

/// Reveal submission data
RevealSubmission {
    target_id: ObjectID,
    data_hash: Digest<Vec<u8>>,            // Hash of data (data hosted off-chain)
    data_url: String,                       // URL where data is hosted for challenge verification
    embedding: Vec<f32>,                    // Pre-computed embedding (verified on challenge only)
    nonce: [u8; 32],
},

/// Claim bond after challenge period
ClaimBond {
    submission_id: ObjectRef,
},
```

Note: Submitter provides `embedding` directly. The transaction executor computes `distance = ||embedding - target.embedding||` and validates `distance < radius`. Full embedding verification (re-running inference) happens only during challenge audit. The model used is the target's assigned model.

### 5.4 Implement Submission Executor

New file `authority/src/execution/submission.rs`:

**CommitSubmission:**
1. Validate target exists
2. If target status is `Filled`:
   - Get existing submission from `target.winning_submission_id`
   - If `existing.reveal_timestamp_ms.is_none()` and `current_timestamp_ms > existing.commit_timestamp_ms + reveal_window_ms`:
     - Slash existing submission's bond to EmissionPool
     - Set target status to `Open`
     - Clear `target.winning_submission_id`
   - Else: reject (target already filled with valid/pending commitment)
3. Validate target status is now `Open`
4. Validate bond >= `submission_min_bond`
5. Create Submission as **shared object** with revealed fields = None, `commit_timestamp_ms` = current timestamp
6. Lock bond
7. Set target status to `Filled { fill_epoch: current_epoch }`
8. Set `target.winning_submission_id` to new submission
9. Generate replacement target (calls target generation)

**RevealSubmission:**
1. Validate submission exists with matching commitment
2. Validate within reveal window: `current_timestamp_ms <= commit_timestamp_ms + reveal_window_ms`
3. Validate `hash(data_hash || embedding || nonce) == commitment`
4. Compute distance = ||embedding - target.embedding|| (L2 norm, simple float computation)
5. Validate distance < target.radius
6. Set revealed fields on submission including `reveal_timestamp_ms`
7. Add to hits[current_epoch]

Note: Embedding is trusted at reveal time. Validators only re-compute embedding during challenge audit to verify it matches what the model produces.

**Reveal Timeout:**
- Handled lazily via CommitSubmission (see above)
- No separate timeout transaction or background cleanup needed
- Expired commitments are cleaned up when the next miner commits to the same target

**ClaimBond:**
1. Validate challenge period ended: `current_epoch > reveal_epoch + 1`
2. Validate submission not in slashed_submissions
3. Transfer bond_amount to miner

### 5.5 Add Protocol Config

```rust
submission_min_bond: Option<u64>,
reveal_window_ms: Option<u64>,      // ~30,000ms (30 seconds)
distance_epsilon: Option<f32>,      // Tolerance for floating-point distance comparisons
```

### 5.6 Add CLI Commands

New file `cli/src/commands/submit.rs`:
- `soma submit commit --target <ID> --data <FILE> --model <MODEL_ID>`
  - Computes embedding, distance, generates commitment, submits
- `soma submit reveal --target <ID> --data <FILE> --model <MODEL_ID> --nonce <NONCE>`
- `soma submit list [--miner <ADDR>] [--target <ID>]`
- `soma submit info <ID>`
- `soma submit claim-bond --submission <ID>`

**Gate:** E2E test for commit → reveal → hit recording

---

## Phase 6: Inference Engine

**Goal:** Shared inference engine for miners, challengers, and validators.

### 6.1 Create Inference Engine Crate

Keep `probes/` as the model architecture crate (transformer implementation).

Create new `inference-engine/` crate that uses probes:
```toml
# inference-engine/Cargo.toml
[dependencies]
probes = { path = "../probes" }
burn = { version = "...", features = ["wgpu"] }
```

### 6.2 Create Engine Trait

New file `inference-engine/src/engine.rs`:
```rust
pub trait InferenceEngineAPI: Send + Sync {
    fn load_model(&self, weights: &[u8]) -> Result<Arc<dyn LoadedModelAPI>>;
    fn compute_embedding(&self, model: &dyn LoadedModelAPI, data: &[u8]) -> Result<Vec<f32>>;
    fn compute_distance(&self, embedding: &[f32], target: &[f32]) -> f32;
}

pub trait LoadedModelAPI: Send + Sync {
    fn model_id(&self) -> &ModelId;
}

pub fn create_inference_engine(arch_version: u64) -> Box<dyn InferenceEngineAPI> {
    match arch_version {
        1 => Box::new(v1::InferenceEngineV1::new()),
        _ => panic!("unsupported architecture version"),
    }
}
```

### 6.3 Create Inference Service

New file `inference-engine/src/service.rs`:
```rust
/// Shared inference service used by CLI and validators
pub struct InferenceService {
    engines: RwLock<BTreeMap<u64, Arc<dyn InferenceEngineAPI>>>,
    loaded_models: RwLock<BTreeMap<ModelId, Arc<dyn LoadedModelAPI>>>,
    model_cache: RwLock<LruCache<ModelId, Vec<u8>>>,  // Cached decrypted weights
}

impl InferenceService {
    pub async fn ensure_model_loaded(&self, model: &Model) -> Result<Arc<dyn LoadedModelAPI>>;
    pub async fn compute_embedding(&self, model_id: &ModelId, data: &[u8]) -> Result<Vec<f32>>;
    pub async fn compute_distance(&self, embedding: &[f32], target: &[f32]) -> f32;
}
```

### 6.4 Determinism Requirements

- All computations must be deterministic across nodes
- Inference in f32, immediately convert to fixed-point for comparisons
- Use Burn WGPU backend with deterministic settings

### 6.5 Add Verification CLI

New file `cli/src/commands/verify.rs`:
- `soma verify --data <FILE> --target <ID> --model <MODEL_ID>`
  - Outputs: embedding, distance to target, whether it's within radius

**Gate:** E2E test for inference determinism across nodes

---

## Phase 7: Challenge System

**Goal:** Challengers can dispute winners, validators audit and vote.

### 7.1 Create Challenge Types

New file `types/src/challenge.rs`:
```rust
pub struct Challenge {
    pub id: ObjectID,
    pub submission_id: ObjectID,
    pub challenger: SomaAddress,
    pub challenge_type: ChallengeType,
    pub challenger_bond: u64,
}

pub enum ChallengeType {
    /// Miner lied about distance
    ScoreFraud,
    /// Data at URL doesn't match commitment
    DataFraud,
}

pub struct ChallengeAuditVote {
    pub challenge_id: ObjectID,
    pub verdict: ChallengeVerdict,
    pub validator: AuthorityName,
    pub signature: AuthoritySignature,
}

pub enum ChallengeVerdict {
    ChallengerWins { reason: ChallengeWinReason },
    MinerWins,
}

pub enum ChallengeWinReason {
    DistanceMismatch { claimed: f32, actual: f32 },
    DataHashMismatch,
    MinerUnresponsive,
}

pub struct CertifiedChallengeResult {
    pub challenge_id: ObjectID,
    pub verdict: ChallengeVerdict,
    pub quorum_signature: AuthorityStrongQuorumSignInfo,
}
```

### 7.2 Add Challenge Transactions

In `types/src/transaction.rs`:
```rust
/// Initiate a challenge against a submission
InitiateChallenge {
    submission_id: ObjectID,
    challenge_type: ChallengeType,
    challenger_bond_coin: ObjectRef,
},

/// Submit certified audit result
ResolveChallengeAudit {
    certified_result: CertifiedChallengeResult,
},
```

### 7.3 Implement Challenge Executor

New file `authority/src/execution/challenge.rs`:

**InitiateChallenge:**
1. Validate submission exists and is in challenge window
   - `reveal_epoch == current_epoch` or `reveal_epoch == current_epoch - 1`
2. Validate no active challenge exists for this submission
3. Validate challenger bond >= `submission.bond_amount / 3`
4. Create Challenge object
5. Lock challenger bond
6. Validators automatically start auditing

**ResolveChallengeAudit:**
1. Verify 2f+1 quorum signature on certified_result
2. If ChallengerWins:
   - Transfer miner bond → challenger
   - Add submission to slashed_submissions
   - Remove from hits[reveal_epoch]
   - Reopen target (status = Open)
3. If MinerWins:
   - Transfer challenger bond → miner

**Challenge Timeout (in ChangeEpoch at end of epoch N+1):**
- For challenges on epoch N submissions that haven't resolved:
  - Return challenger bond
  - Miner remains winner

### 7.4 Implement Validator Audit Service

New file `authority/src/challenge_audit.rs`:
```rust
pub struct ChallengeAuditService {
    inference_service: Arc<InferenceService>,
    state: Arc<AuthorityState>,
    http_client: reqwest::Client,
}

impl ChallengeAuditService {
    pub async fn audit_challenge(&self, challenge: &Challenge) -> ChallengeAuditVote {
        let submission = self.get_submission(&challenge.submission_id);

        // 1. Download data from submission's data_url
        let data = match self.download_data(&submission.data_url).await {
            Ok(d) => d,
            Err(_) => return vote(ChallengerWins(MinerUnresponsive)),
        };

        // 2. Check DataFraud
        if hash(&data) != submission.data_hash {
            return vote(ChallengerWins(DataHashMismatch));
        }

        // 3. Check ScoreFraud
        let model = self.inference_service.ensure_model_loaded(&submission.model_id).await?;
        let embedding = self.inference_service.compute_embedding(&model, &data)?;
        let target = self.get_target(&submission.target_id);
        let actual_distance = self.inference_service.compute_distance(&embedding, &target.embedding);

        if (actual_distance - submission.distance).abs() > distance_epsilon {
            return vote(ChallengerWins(DistanceMismatch {
                claimed: submission.distance,
                actual: actual_distance
            }));
        }

        vote(MinerWins)
    }
}
```

### 7.5 Implement Challenge Vote Exchange

Extend validator P2P:
- New RPC method: `BroadcastChallengeVote(ChallengeAuditVote)`
- Aggregate votes in `ChallengeAggregator`
- When 2f+1 votes agree → create `CertifiedChallengeResult` → submit `ResolveChallengeAudit` tx

### 7.6 Wire into Node

In `node/src/lib.rs`:
- Start `ChallengeAuditService` for validators
- Subscribe to new Challenge objects
- Trigger audit when challenge is created

### 7.7 Add Protocol Config

```rust
challenge_bond_ratio_bps: Option<u64>,     // 3333 = 1/3 of miner bond
challenge_audit_timeout_ms: Option<u64>,   // How long validators wait for data
```

### 7.8 Add CLI Commands

New file `cli/src/commands/challenge.rs`:
- `soma challenge initiate --submission <ID> --type <score-fraud|data-fraud>`
- `soma challenge list [--submission <ID>] [--challenger <ADDR>]`
- `soma challenge info <ID>`

**Gate:** E2E test for full challenge lifecycle

---

## Phase 8: Reward Distribution & Difficulty Adjustment

**Goal:** Winners receive rewards at epoch end, difficulty adjusts based on hit rate.

### 8.1 Implement Reward Distribution in ChangeEpoch

In `authority/src/execution/change_epoch.rs`, at end of epoch N+1:

```rust
// 1. Close challenge window for epoch N
let confirmed_hits: Vec<_> = hits[N].iter()
    .filter(|s| !slashed_submissions.contains(s))
    .collect();
let num_confirmed_hits = confirmed_hits.len();

// 2. Calculate per-hit reward from epoch N's emissions
// Note: Epoch N emissions were calculated during epoch N's ChangeEpoch
let epoch_n_target_emissions = get_stored_target_emissions(N);  // 80% of epoch N emissions
let per_hit_reward = if num_confirmed_hits > 0 {
    epoch_n_target_emissions / num_confirmed_hits as u64
} else {
    0  // No hits = emissions returned to pool
};

// 3. Distribute rewards for each confirmed hit
for submission_id in confirmed_hits {
    let submission = get_submission(&submission_id);
    let target = get_target(&submission.target_id);
    let model = get_model(&submission.model_id);

    let miner_share = per_hit_reward * miner_reward_share_bps / 10000;  // 50%
    let model_share = per_hit_reward * model_reward_share_bps / 10000;  // 30%
    // Remaining 20% already distributed to validators in epoch N

    create_coin(submission.miner, miner_share);
    create_coin(model.owner, model_share);
    model.total_rewards_earned += model_share;

    // Confirm embedding addition to model index
    model_index.add_embedding(submission.embedding.clone(), submission.model_id);

    // Track hit time for difficulty adjustment
    let hit_time = submission.reveal_timestamp_ms.unwrap() - target.generation_timestamp_ms;
    difficulty_stats.total_hit_time_ms += hit_time;
    difficulty_stats.confirmed_hit_count += 1;
}

// 4. If no confirmed hits, return target emissions to pool
if num_confirmed_hits == 0 {
    emission_pool.balance += epoch_n_target_emissions;
}

// 5. Clean up expired challenges (return challenger bonds)
for (submission_id, challenge_id) in pending_challenges.iter() {
    let submission = get_submission(submission_id);
    if submission.reveal_epoch == Some(N) {
        let challenge = get_challenge(challenge_id);
        // Challenge window expired without resolution - return challenger bond
        create_coin(challenge.challenger, challenge.challenger_bond);
        pending_challenges.remove(submission_id);
    }
}

// 6. Clean up old hit tracking
hits.remove(&N);
```

### 8.2 Implement Difficulty Adjustment

In `authority/src/execution/change_epoch.rs`:

```rust
// Use accumulated difficulty stats from reward distribution
if difficulty_stats.confirmed_hit_count > 0 {
    let avg_hit_time_ms = difficulty_stats.total_hit_time_ms / difficulty_stats.confirmed_hit_count;

    // Adjust radius
    if avg_hit_time_ms < target_hit_time_ms {
        target_state.current_radius *= (1.0 - radius_adjustment_rate);  // Harder
    } else if avg_hit_time_ms > target_hit_time_ms {
        target_state.current_radius *= (1.0 + radius_adjustment_rate);  // Easier
    }
    target_state.current_radius = target_state.current_radius.clamp(min_radius, max_radius);
}

// Reset stats for next epoch
difficulty_stats = DifficultyStats::default();
```

**Gate:** E2E test for reward distribution, difficulty adjustment

---

## Phase 9: Error Handling

### 9.1 Add ExecutionFailureStatus Variants

In `types/src/effects.rs`:
```rust
pub enum ExecutionFailureStatus {
    // Model errors
    InsufficientRegistrationFee { required: u64, provided: u64 },
    ModelWeightsTooLarge { size: u64, max: u64 },
    ModelArchitectureMismatch { expected: u64, actual: u64 },
    ModelNotActive { model_id: ModelId },
    InvalidThresholdShares { reason: String },
    InvalidCapabilityEmbedding { reason: String },

    // Target errors
    TargetNotOpen { target_id: ObjectID },

    // Submission errors
    ModelNotInTarget { model_id: ModelId, target_id: ObjectID },
    DistanceExceedsRadius { distance: f32, radius: f32 },
    DistanceMismatch { claimed: f32, actual: f32 },
    RevealWindowExpired { commit_block: u64, current_block: u64 },
    CommitmentMismatch,
    InsufficientBond { required: u64, provided: u64 },

    // Challenge errors
    SubmissionNotInChallengeWindow { reveal_epoch: EpochId, current_epoch: EpochId },
    ChallengeAlreadyExists { challenge_id: ObjectID },
    InsufficientChallengerBond { required: u64, provided: u64 },

    // Bond errors
    BondLocked { reveal_epoch: EpochId, current_epoch: EpochId },
    BondAlreadySlashed { submission_id: ObjectID },

    // Audit errors
    InvalidChallengeQuorum { required: u64, provided: u64 },
}
```

---

## Phase 10: RPC & SDK Updates

### 10.1 Add RPC Index Tables

In `authority/src/rpc_index.rs`:
```rust
struct IndexStoreTables {
    // Models
    model_by_owner: DBMap<ModelOwnerKey, ModelId>,

    // Targets
    target_by_status: DBMap<TargetStatusKey, ObjectID>,

    // Submissions
    submission_by_target: DBMap<SubmissionTargetKey, ObjectID>,
    submission_by_miner: DBMap<SubmissionMinerKey, ObjectID>,

    // Challenges
    challenge_by_submission: DBMap<ChallengeSubmissionKey, ObjectID>,
    challenge_by_challenger: DBMap<ChallengeChallengerKey, ObjectID>,
}
```

### 10.2 Update RPC Protos

In `rpc/src/proto/soma.proto`:
```protobuf
// Model queries
rpc GetModel(GetModelRequest) returns (GetModelResponse);
rpc ListModels(ListModelsRequest) returns (ListModelsResponse);

// Target queries
rpc GetTarget(GetTargetRequest) returns (GetTargetResponse);
rpc ListOpenTargets(ListOpenTargetsRequest) returns (ListOpenTargetsResponse);

// Submission queries
rpc GetSubmission(GetSubmissionRequest) returns (GetSubmissionResponse);
rpc ListSubmissionsByTarget(ListSubmissionsByTargetRequest) returns (ListSubmissionsByTargetResponse);
rpc ListSubmissionsByMiner(ListSubmissionsByMinerRequest) returns (ListSubmissionsByMinerResponse);

// Challenge queries
rpc GetChallenge(GetChallengeRequest) returns (GetChallengeResponse);
rpc ListChallenges(ListChallengesRequest) returns (ListChallengesResponse);
```

### 10.3 Update SDK

In `sdk/src/lib.rs`:
```rust
// Model operations
pub async fn commit_model(&self, ...) -> Result<ModelId>;
pub async fn get_model(&self, model_id: &ModelId) -> Result<Model>;
pub async fn list_models(&self, filter: ModelFilter) -> Result<Vec<Model>>;

// Target operations
pub async fn get_target(&self, target_id: &ObjectID) -> Result<Target>;
pub async fn list_open_targets(&self) -> Result<Vec<Target>>;

// Submission operations
pub async fn commit_submission(&self, ...) -> Result<ObjectID>;
pub async fn reveal_submission(&self, ...) -> Result<()>;
pub async fn claim_bond(&self, submission_id: &ObjectID) -> Result<()>;

// Challenge operations
pub async fn initiate_challenge(&self, ...) -> Result<ObjectID>;
```

**Gate:** SDK integration tests

---

## Phase 11: Polish & Testnet

### 11.1 Comprehensive Testing

| Test File | Coverage |
|-----------|----------|
| `threshold_reveal.rs` | Share generation, encryption, EOP broadcast, combination |
| `model_lifecycle.rs` | Commit, epoch reveal, index addition |
| `target_generation.rs` | Seed determinism, kNN selection, replacement on fill |
| `submission_flow.rs` | Commit, reveal, timeout, hit recording |
| `challenge_flow.rs` | Both challenge types, audit voting, resolution |
| `reward_distribution.rs` | Miner/model/validator split, slashed handling |
| `difficulty_adjustment.rs` | Radius changes based on hit rate |
| `bond_mechanics.rs` | Lock, slash, claim |
| `emission_schedule.rs` | Step-decay, allocation split |

### 11.2 Documentation

- Update CLAUDE.md with new architecture
- CLI help text for all new commands
- SDK examples for full flow

### 11.3 Testnet Deployment

- Genesis with initial models
- Monitoring for challenge throughput
- Validator audit performance testing

---

## Quick Reference: Files to Delete

| Location | Files |
|----------|-------|
| Workspace | `encoder/`, `encoder-validator-api/`, `test-encoder-cluster/`, `intelligence/`, `vdf/`, `arrgen/` |
| `types/src/` | `encoder_committee.rs`, `encoder_info.rs`, `encoder_validator/`, `shard.rs`, `report.rs`, `submission.rs`, `shard_networking/`, `shard_crypto/`, `shard_verifier.rs`, `entropy.rs`, `evaluation.rs` |
| `types/src/config/` | `encoder_config.rs` |
| `types/src/system_state/` | `encoder.rs`, `shard.rs` |
| `authority/src/` | `encoder_client.rs` |
| `authority/src/execution/` | `encoder.rs`, `shard.rs` |
| `cli/src/commands/` | `encoder.rs` |
| Tests | `types/src/system_state/unit_tests/encoder_staking.rs`, `e2e-tests/tests/encoder_committee_tests.rs` |

## Quick Reference: Files to Create

| Crate | File | Contents |
|-------|------|----------|
| types | `threshold_bls.rs` | EncryptedKeyShare, ThresholdKeyShares, DecryptedKeyShare |
| types | `crypto/threshold.rs` | ThresholdBls implementation |
| types | `model.rs` | Model, ModelId, ModelRegistry |
| types | `knn_index.rs` | kNNIndex for model selection |
| types | `target.rs` | Target, TargetStatus, TargetState |
| types | `submission.rs` | Submission (new version with hash-commit-reveal) |
| types | `challenge.rs` | Challenge, ChallengeType, ChallengeVerdict |
| authority | `execution/model.rs` | Model tx executor |
| authority | `execution/target.rs` | Target generation |
| authority | `execution/submission.rs` | Submission tx executor |
| authority | `execution/challenge.rs` | Challenge tx executor |
| authority | `challenge_audit.rs` | Validator audit service |
| inference-engine | `lib.rs` | Crate root |
| inference-engine | `engine.rs` | InferenceEngineAPI trait |
| inference-engine | `service.rs` | InferenceService |
| cli | `commands/model.rs` | Model CLI |
| cli | `commands/submit.rs` | Submission CLI |
| cli | `commands/verify.rs` | Verification CLI |
| cli | `commands/challenge.rs` | Challenge CLI |

## Quick Reference: Files to Modify

| Crate | File | Changes |
|-------|------|---------|
| types | `transaction.rs` | Remove encoder txs, add Model/Submission/Challenge txs, extend ChangeEpoch args |
| types | `consensus/mod.rs` | Extend EndOfPublish to carry decrypted shares |
| types | `system_state/mod.rs` | Add ModelRegistry, TargetState, HitTracking, DifficultyStats, remove encoder fields |
| types | `system_state/emission.rs` | Step-decay fields, allocation split |
| types | `effects.rs` | New ExecutionFailureStatus variants |
| types | `finality.rs` | No changes needed (already generic) |
| types | `lib.rs` | Export new modules |
| authority | `execution/mod.rs` | Add model/target/submission/challenge executors |
| authority | `execution/change_epoch.rs` | Threshold reveals, reward distribution, difficulty adjustment, challenge timeout |
| authority | `checkpoints/checkpoint_executor/mod.rs` | Collect shares from EOP, pass to ChangeEpoch |
| node | `lib.rs` | Challenge audit wiring |
| protocol-config | `lib.rs` | All new parameters |
| rpc | `proto/soma.proto` | New query methods |
| cli | `commands/mod.rs` | Register new command modules |
| sdk | `lib.rs` | New client methods |
| workspace | `Cargo.toml` | Remove encoder crates, add inference-engine, add hnswlib-rs |

## Quick Reference: Protocol Config Parameters

```rust
// Emission
initial_emission_per_epoch: Option<u64>,
emission_period_length: Option<u64>,
emission_decrease_rate_bps: Option<u16>,
validator_reward_share_bps: Option<u64>,  // 2000 = 20%
miner_reward_share_bps: Option<u64>,      // 5000 = 50%
model_reward_share_bps: Option<u64>,      // 3000 = 30%

// Model
model_registration_fee: Option<u64>,
model_architecture_version: Option<u64>,
model_max_weight_size: Option<u64>,
embedding_dim: Option<u64>,

// Target
k_models_per_target: Option<u64>,
target_generation_budget_per_epoch: Option<u64>,
initial_radius: Option<f32>,
min_radius: Option<f32>,
max_radius: Option<f32>,
target_hit_time_ms: Option<u64>,
radius_adjustment_rate: Option<f32>,

// Submission
submission_min_bond: Option<u64>,
reveal_window_ms: Option<u64>,
distance_epsilon: Option<f32>,

// Challenge
challenge_bond_ratio_bps: Option<u64>,
challenge_audit_timeout_ms: Option<u64>,
```

## Quick Reference: New Dependencies

| Crate | Dependency | Purpose |
|-------|------------|---------|
| types | `hnswlib-rs` | kNN index for model selection (supports incremental insert/delete) |
| types | `fastcrypto` | Threshold BLS operations |
| inference-engine | `burn` | Neural network inference |
| inference-engine | `probes` | Transformer model architecture |
