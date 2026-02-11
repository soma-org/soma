# SOMA Redesign: Implementation Plan

**Key Context:** Chain is not live. Delete freely, no migration needed.

---

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Core Removals | ✅ Complete | Encoder infrastructure removed |
| Phase 2: Step-Decay Emissions | ⏳ Pending | Logarithmic emission decay |
| Phase 3: Model Registry | ✅ Complete | Commit-reveal flow, staking, CLI, RPC |
| Phase 4: Target Generation | ✅ Complete | Continuous targets, difficulty adjustment |
| Phase 5: Data Submission | ✅ Complete | SubmitData, ClaimRewards, bond mechanics |
| Phase 6: Inference Engine | 🔄 In Progress | CompetitionAPI trait defined, mocks implemented |
| Phase 7: Challenge System | ✅ Complete | Tally-based voting, AuditService, 14 E2E tests |
| Phase 8: Reward Distribution | ⏳ Pending | Epoch-end reward distribution |
| Phase 9: Error Handling | ⏳ Pending | ExecutionFailureStatus variants |
| Phase 10: RPC & SDK Updates | ⏳ Pending | Index tables, SDK methods |
| Phase 11: Polish & Testnet | ⏳ Pending | Final testing, documentation |

### E2E Test Summary

| Test Suite | Tests | Status |
|------------|-------|--------|
| target_tests | 6 | ✅ Passing |
| challenge_tests | 14 | ✅ Passing |
| model_tests | 2 | ✅ Passing |
| **Total** | **22** | **✅ All Passing** |

### Next Steps

1. **Phase 6: Real CompetitionAPI** — Replace `MockCompetitionAPI` with actual inference using `probes/` crate
2. **CLI UX improvements** — Better error messages, progress indicators, output formatting
3. **Additional chain-side tests** — More edge cases for targets, submissions, epoch boundary logic
4. **Phase 2: Emissions** — Step-decay schedule (can be done in parallel)

---

## Key Technical Concepts

### BCS-Compatible Float Types

BCS (Binary Canonical Serialization) does not support f32 directly. We use wrapper types that serialize floats as raw bytes (little-endian IEEE 754).

| Type | Location | Purpose |
|------|----------|---------|
| `BcsF32` | `protocol-config/src/tensor.rs` | Single f32 values in ProtocolConfig |
| `SomaTensor` | `protocol-config/src/tensor.rs` | Multi-dimensional f32 arrays (embeddings, distances) |

**BcsF32**: Wraps a single f32 value for use in protocol config fields (e.g., distance thresholds).

```rust
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, JsonSchema)]
pub struct BcsF32(pub f32);

impl Serialize for BcsF32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.to_le_bytes().serialize(serializer)  // 4 bytes, little-endian
    }
}
```

**SomaTensor**: Wraps Burn's `TensorData` with Hash implementation for transaction serialization.

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct SomaTensor(pub TensorData);

impl Hash for SomaTensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.shape.hash(state);
        self.0.as_bytes().hash(state);  // Deterministic for identical f32 values
    }
}
```

**Re-export**: `types/src/tensor.rs` re-exports `SomaTensor` from protocol-config to avoid circular dependencies.

### Tolerance Checking

For off-chain fraud detection, validators use Burn's `Tolerance::permissive()`:
- **Relative tolerance**: 1% (0.01)
- **Absolute tolerance**: 0.01
- **Formula**: `|x - y| < max(0.01 * (|x + y|), 0.01)`

This is more lenient than `Tensor::all_close()` defaults (1e-5/1e-8), appropriate for GPU variance across different hardware.

```rust
fn is_within_tolerance(computed: &TensorData, claimed: &TensorData) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        computed.assert_approx_eq::<f32>(claimed, Tolerance::permissive());
    })).is_ok()
}
```

### CompetitionAPI

The inference interface for validators, defined in `runtime/src/lib.rs`:

```rust
#[async_trait]
pub trait CompetitionAPI: Send + Sync + 'static {
    async fn run(&self, input: CompetitionInput) -> RuntimeResult<CompetitionOutput>;
}

pub struct CompetitionInput {
    data: Manifest,
    models: Vec<(ModelId, Manifest)>,
    target: TensorData,
}

pub struct CompetitionOutput {
    winner: ModelId,
    embedding: TensorData,
    distance: TensorData,
}
```

**Mock implementations** (in `authority/src/audit_service.rs`):
- `MockCompetitionAPI`: Returns matching results (no fraud detected)
- `MockFraudCompetitionAPI`: Returns error simulating data unavailability (fraud detected)

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
- **Submissions**: Single-transaction with bond (commit-reveal deferred)
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
   - Submit data + embedding + distance_score + bond
   - First valid submission wins, target marked Filled
   - Replacement target spawned automatically

4. **Challenge (epoch N through N+1):**
   - Anyone can challenge with bond (1/3 of miner bond)
   - Validators audit via CompetitionAPI, submit individual report transactions
   - 2f+1 stake-weighted quorum resolves challenge
   - Successful challenge → miner slashed, rewards redistributed
   - ClaimRewards/ClaimChallengeBond distributes funds after challenge window

---

## Phase 1: Core Removals — COMPLETE ✓

**Goal:** Codebase compiles without encoder infrastructure.

Removed: encoder crates, VDF, shard types, encoder transactions, encoder executors, encoder CLI commands.

**Gate:** ✓ `cargo build --workspace && cargo test -p types -p authority -p node`

---

## Phase 2: Step-Decay Emissions

**Goal:** Replace linear emissions with logarithmic decay.

**Status:** Pending

### Changes Required

1. **EmissionPool** (`types/src/system_state/emission.rs`):
   - Add `subsidy_period_length`, `subsidy_decrease_rate_bps`
   - Update `advance_epoch()` with decay logic

2. **Emission Allocation**:
   - 20% to validators (proportional to stake)
   - 80% to target winner pool (50% miner / 30% model)

3. **Protocol Config Parameters**:
   - `initial_emission_per_epoch`, `emission_period_length`, `emission_decrease_rate_bps`
   - `validator_reward_share_bps`, `miner_reward_share_bps`, `model_reward_share_bps`

**Gate:** Unit tests for step-decay schedule, allocation split

---

## Phase 3: Model Registry with Staking — COMPLETE ✓

**Goal:** Anyone can register models with staking, using simple commit-reveal flow.

**Status:** Fully implemented. 18 unit tests, 2 E2E tests passing.

**Key Files:**
- `types/src/model.rs` — Model, ModelId, PendingModelUpdate
- `types/src/system_state/model_registry.rs` — ModelRegistry
- `authority/src/execution/model.rs` — 9 transaction executors
- `cli/src/commands/model.rs` — Full CLI support

---

## Phase 4: Target Generation — COMPLETE ✓

**Goal:** Continuous target generation with model selection at genesis and epoch boundaries.

**Status:** Fully implemented. 17 unit tests, 6 E2E tests passing.

### Key Design Decisions

- **Hit-rate EMA for difficulty adjustment**: Uses `hits_this_epoch / targets_generated_this_epoch` with EMA across epochs
- **Uniform random model selection**: Stake-weighting deferred to future phase
- **Deterministic embedding via arrgen**: Uses `arrgen::normal_array` with checkpoint-derived seed
- **f32 distance thresholds**: Stored as `SomaTensor::scalar()` for BCS compatibility

**Key Files:**
- `types/src/target.rs` — Target, TargetId, TargetStatus, generate_target()
- `types/src/system_state/target_state.rs` — TargetState with SomaTensor distance_threshold
- `protocol-config/src/lib.rs` — Distance threshold parameters as `Option<BcsF32>`

**Gate:** ✓ `RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test target_tests` — 6 tests passing

---

## Phase 5: Data Submission — COMPLETE ✓

**Goal:** Single-transaction submission flow with bond, rewards, and permissionless claiming.

**Status:** Fully implemented. 14 unit tests, included in target E2E tests.

### Key Design Decisions

- **Single-transaction submission** (no commit-reveal): Simplifies flow, front-running protection deferred
- **f32 distance scores**: Stored as `SomaTensor::scalar()` for consistency with CompetitionAPI
- **Permissionless claiming**: Anyone can claim rewards after challenge window
- **Bond scales with data size**: `bond = submission_bond_per_byte * data_size`

**Key Files:**
- `types/src/submission.rs` — Submission with SomaTensor embedding and distance_score
- `authority/src/execution/submission.rs` — SubmitData, ClaimRewards executors
- `cli/src/commands/submit.rs`, `cli/src/commands/claim.rs` — CLI support

**Gate:** ✓ `RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test target_tests` — 6 tests passing

---

## Phase 6: Inference Engine — IN PROGRESS

**Goal:** Shared inference engine for miners, challengers, and validators.

**Status:** CompetitionAPI trait defined in `runtime/`. Mock implementations in AuditService. Real inference pending.

### Completed

- **CompetitionAPI trait** (`runtime/src/lib.rs`): Defines input/output for inference
- **CompetitionInput/Output**: Uses Burn's TensorData for embeddings and distances
- **MockCompetitionAPI**: Returns matching results for "no fraud" path
- **MockFraudCompetitionAPI**: Returns DataNotAvailable error for "fraud detected" path
- **AuditService integration**: Uses CompetitionAPI for fraud verification

### Remaining Work

1. **Real CompetitionV1 implementation**:
   - Download model weights from manifests (cached)
   - Download input data from data_manifest
   - Verify data hash matches commitment
   - Run inference with each model using `probes/` crate
   - Determine winning model (best distance to target)

2. **Determinism requirements**:
   - Consistent f32 inference across validators
   - Use same backend (CPU/GPU) configuration

3. **Verification CLI**:
   - `soma verify --data <FILE> --target <ID> --model <MODEL_ID>`

**Gate:** E2E test for inference determinism across nodes

---

## Phase 7: Challenge System — COMPLETE ✓

**Goal:** Challengers can dispute winners, validators audit and vote.

**Status:** Fully implemented. 14 E2E tests passing.

### Key Design Decisions

- **Tally-based voting**: Validators submit individual ReportSubmission/ReportChallenge transactions
- **Fraud-only challenges**: Availability issues handled via ReportSubmission without InitiateChallenge
- **2f+1 stake-weighted quorum**: `QUORUM_THRESHOLD = 6667` (66.67% of voting power)
- **Tolerance::permissive()**: 1% relative, 0.01 absolute tolerance for distance comparisons

### Key Files

- `types/src/challenge.rs` — Challenge, ChallengeId, ChallengeStatus
- `authority/src/execution/challenge.rs` — Challenge transaction executors
- `authority/src/audit_service.rs` — AuditService with CompetitionAPI integration
- `e2e-tests/tests/challenge_tests.rs` — 14 comprehensive tests

### AuditService Flow

```
Challenge Created → AuditService receives via channel
    ↓
CompetitionAPI.run(input) → Download data, run inference
    ↓
Compare results:
  - DataNotAvailable → FRAUD (miner responsible for availability)
  - DataHashMismatch → FRAUD (miner submitted wrong data)
  - Wrong model → FRAUD (miner didn't use winning model)
  - Distance outside tolerance → FRAUD
  - Otherwise → NO FRAUD
    ↓
Submit Report:
  - FRAUD → ReportSubmission (with challenger attribution)
  - NO FRAUD → ReportChallenge (challenger was wrong)
```

**Gate:** ✓ `RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test challenge_tests` — 14 tests passing

---

## Phase 8: Reward Distribution & Difficulty Adjustment

**Goal:** Winners receive rewards at epoch end, difficulty adjusts based on hit rate.

**Status:** Pending

### Implementation in ChangeEpoch

1. Close challenge window for epoch N
2. Calculate per-hit reward from epoch N's emissions
3. Distribute rewards (50% miner, 30% model owner)
4. Clean up expired challenges (return bonds)
5. Adjust difficulty based on hit-rate EMA

---

## Phase 9: Error Handling

**Goal:** Comprehensive ExecutionFailureStatus variants.

**Status:** Pending. Add variants for model, target, submission, challenge, and bond errors.

---

## Phase 10: RPC & SDK Updates

**Goal:** Index tables for efficient queries, SDK client methods.

**Status:** Pending. Basic RPC handlers exist, need index tables for scalable queries.

---

## Phase 11: Polish & Testnet

**Goal:** Final testing, documentation, testnet deployment.

**Status:** Pending.

---

## Quick Reference: Files Created

| Crate | File | Status |
|-------|------|--------|
| protocol-config | `tensor.rs` | ✓ BcsF32 + SomaTensor |
| types | `model.rs` | ✓ Model, ModelId |
| types | `target.rs` | ✓ Target, TargetStatus |
| types | `submission.rs` | ✓ Submission with SomaTensor |
| types | `challenge.rs` | ✓ Challenge, ChallengeStatus |
| types | `tensor.rs` | ✓ Re-exports SomaTensor |
| types | `system_state/model_registry.rs` | ✓ ModelRegistry |
| types | `system_state/target_state.rs` | ✓ TargetState with SomaTensor |
| authority | `execution/model.rs` | ✓ Model executor |
| authority | `execution/submission.rs` | ✓ Submission executor |
| authority | `execution/challenge.rs` | ✓ Challenge executor |
| authority | `audit_service.rs` | ✓ AuditService + CompetitionAPI mocks |
| runtime | `lib.rs` | ✓ CompetitionAPI trait |
| runtime | `competition/` | ✓ Competition module |
| cli | `commands/model.rs` | ✓ Model CLI |
| cli | `commands/submit.rs` | ✓ Submit CLI |
| cli | `commands/claim.rs` | ✓ Claim CLI |
| cli | `commands/target.rs` | ✓ Target CLI |
| cli | `commands/challenge.rs` | ✓ Challenge CLI |
| e2e-tests | `tests/model_tests.rs` | ✓ 2 tests |
| e2e-tests | `tests/target_tests.rs` | ✓ 6 tests |
| e2e-tests | `tests/challenge_tests.rs` | ✓ 14 tests |
| inference-engine | `lib.rs` | ⏳ Pending |

## Quick Reference: Protocol Config Parameters

```rust
// Distance thresholds (BcsF32 for BCS compatibility)
target_initial_distance_threshold: Option<BcsF32>,  // Default: 0.5
target_max_distance_threshold: Option<BcsF32>,      // Default: 1.0
target_min_distance_threshold: Option<BcsF32>,      // Default: 0.1

// Target generation
k_models_per_target: Option<u64>,
target_generation_budget_per_epoch: Option<u64>,
target_hit_rate_target_bps: Option<u64>,
target_difficulty_adjustment_rate_bps: Option<u64>,

// Submission
submission_bond_per_byte: Option<u64>,

// Challenge
challenge_bond_ratio_bps: Option<u64>,

// Rewards
target_miner_reward_share_bps: Option<u64>,   // 5000 = 50%
target_model_reward_share_bps: Option<u64>,   // 3000 = 30%
target_claimer_incentive_bps: Option<u64>,    // 100 = 1%
```

## Quick Reference: Dependencies

| Crate | Dependency | Purpose |
|-------|------------|---------|
| protocol-config | `burn` | TensorData for SomaTensor |
| protocol-config | `bcs` | Binary serialization |
| types | `protocol-config` | Re-exports SomaTensor |
| authority | `burn` | Tolerance checking |
| authority | `runtime` | CompetitionAPI trait |
| runtime | `burn` | TensorData, inference |
