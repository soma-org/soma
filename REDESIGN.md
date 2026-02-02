# SOMA Redesign Spec

**Context:** The chain is not live, so we can delete freely without migration concerns.

---

## Overview

The new system is a continuous mining competition where data miners race to fill targets. Targets are generated deterministically and select a single model per target. First valid commitment wins. Challenges extend through the following epoch.

---

## Model System

### Model Registration with Staking

Models must stake to participate, with stake serving as:
1. **Reputation proxy**: Higher stake = more trusted model
2. **Minimum requirement**: Models need min stake to be eligible for target assignment
3. **Slash collateral**: Stake can be slashed if model weights unavailable (triggered by validator audit)

### Model Commit-Reveal Flow

**Epoch N (Commit):**
- Model owner submits `CommitModel` transaction with:
  - `weights_url_commitment`: hash of encrypted weights URL
  - `weight_commitment`: hash of actual model weights
  - Stake (transferred to model's staking pool)
- Model is created in `pending` state
- Model can receive delegated stake from other users

**Epoch N+1 (Reveal):**
- Model owner submits `RevealModel` transaction with:
  - `encrypted_weights_url`: actual URL where encrypted weights are hosted
  - `decryption_key`: symmetric key to decrypt the weights
- Validators can verify `hash(encrypted_weights_url) == weights_url_commitment`
- Model transitions to `active` state and becomes eligible for target assignment

**Failed Reveal:**
- If model owner doesn't reveal by end of epoch N+1:
  - Model stake gets slashed (configurable rate, e.g., 50%)
  - Slashed amount returned to emission pool
  - Model removed from pending set

### Model Updates

Models can update their weights using the same commit-reveal flow:
- `CommitModelUpdate` in epoch N (commits new weights while keeping old active)
- `RevealModelUpdate` in epoch N+1 (activates new weights)
- Failed update reveal: update cancelled, no slash (original weights remain active)

### Model Availability Audits

Validators can be triggered to audit model availability:

1. **Challenger reports unavailable model**: Any user can submit `ReportModelUnavailable`
2. **Validator audit**: Validators attempt to download and verify model weights
3. **Audit result**:
   - If weights unavailable: Model gets tallied (strike count incremented)
   - If weights available: Challenger loses their challenge bond
4. **Tally threshold**: Models with tallies >= threshold in an epoch get removed and slashed

Tallies reset at epoch boundary.

### Model Object

```rust
Model {
    id: ModelId,
    owner: Address,
    architecture_version: u64,

    // Commit state
    weights_url_commitment: Hash,      // hash(encrypted_weights_url)
    weight_commitment: Hash,           // hash(decrypted_weights)
    commit_epoch: u64,

    // Reveal state (None if pending)
    encrypted_weights_url: Option<String>,
    decryption_key: Option<[u8; 32]>,

    // Staking
    staking_pool: StakingPool,

    // Pending update (if any)
    pending_update: Option<PendingModelUpdate>,

    // Audit state
    tally_count: u64,  // Resets each epoch

    // Lifetime stats
    total_rewards_earned: u64,
}

struct PendingModelUpdate {
    weights_url_commitment: Hash,
    weight_commitment: Hash,
    commit_epoch: u64,
}
```

### ModelRegistry in SystemState

```rust
ModelRegistry {
    active_model_ids: BTreeSet<ModelId>,
    pending_model_ids: BTreeSet<ModelId>,  // Committed, awaiting reveal
    staking_pool_mappings: BTreeMap<ObjectID, ModelId>,  // pool_id -> model

    // For weighted model selection
    total_model_stake: u64,
}
```

### Model Selection for Targets

Models are selected for targets using stake-weighted random selection:
- Seed: `checkpoint_digest || target_index`
- Weight: Model's staking pool balance
- Higher stake = higher probability of selection

This replaces the kNN index approach with a simpler stake-weighted lottery.

### CommitModel Execution

1. Validate stake >= `model_min_stake`
2. Validate architecture matches `current_architecture_version`
3. Create Model object with reveal fields = None
4. Initialize staking pool with provided stake
5. Add to `pending_model_ids`

### RevealModel Execution

1. Validate model exists in `pending_model_ids`
2. Validate within reveal window: `current_epoch <= commit_epoch + 1`
3. Validate `hash(encrypted_weights_url) == weights_url_commitment`
4. Set reveal fields on model
5. Move from `pending_model_ids` to `active_model_ids`
6. Model becomes eligible for target selection

### Model Reveal Timeout (in ChangeEpoch)

For each model in `pending_model_ids` where `commit_epoch < current_epoch`:
- Slash model stake by `model_reveal_slash_rate_bps`
- Return slashed amount to emission pool
- Remove from `pending_model_ids`
- Move staking pool to inactive (delegators can withdraw)

---

## Target Generation

Targets are generated continuously. When a target is filled, a new one is immediately generated to replace it.

### Generation Process

- **Seed**: `checkpoint_contents_digest || target_index || nonce` (unpredictable)
  - For replacement targets, use the checkpoint containing the fill transaction
- **Embedding**: `E = deterministic_gaussian(seed, embedding_dim)`
- **Model selection**: Single model selected via stake-weighted random sampling
- **Radius**: `R = current_difficulty_radius` (from rolling difficulty)
- **Rewards**: Distributed from epoch emissions at epoch N+1 boundary (emissions / num_confirmed_hits)

### Target Object

```rust
Target {
    id: TargetId,
    embedding: Vector<f32>,
    model_id: ModelId,        // Single model selected via stake-weighted sampling
    radius: f32,
    generation_epoch: u64,
    generation_timestamp_ms: u64,  // For difficulty adjustment

    // State
    status: TargetStatus,  // Open, Filled, Challenged
    winning_submission_id: Option<SubmissionId>,
}

enum TargetStatus {
    Open,
    Filled { fill_epoch: u64 },
    Challenged,
}
```

Note: Targets no longer have individual `reward_pool` fields. Rewards are calculated at epoch N+1 boundary as `epoch_emissions * 80% / num_confirmed_hits`.

### Active Target Count

The number of active targets is implicitly controlled by the difficulty mechanism:

- Tighter radius → targets take longer to fill → more concurrent open targets
- Looser radius → targets fill faster → fewer concurrent open targets

The system maintains a `target_generation_budget` per epoch. New targets spawn when previous ones are filled, drawing from this budget.

---

## Mining & Submission

### Mining Flow

1. Miner observes open target T with embedding E, radius R, and assigned model
2. Miner checks their data portfolio for candidates within distance R of E
3. For each candidate data D:
   - Download the assigned model locally (if not cached)
   - Embed D with the model
   - Compute `distance = ||embed(D, model) - E||`
   - If `distance < R`: valid hit candidate
4. Submit commitment for best candidate

### Submission Commit

Submissions use simple hash-commit-reveal:

```rust
CommitSubmission {
    target_id: TargetId,
    commitment: Hash,  // hash(data_hash || embedding || model_id || nonce)
    bond: Coin,
}
```

**CommitSubmission Execution:**

1. Validate target exists and status is `Open`
2. Validate bond >= `submission_min_bond`
3. Record commitment with consensus-assigned sequence
4. Lock bond
5. **First commitment wins provisionally** (by consensus ordering)
6. Set target status to `Filled { fill_epoch: current_epoch }`

**Note on concurrent commits**: When multiple miners commit in the same consensus batch, transactions are ordered deterministically by consensus commit index. Shared object (Target) versions are pre-assigned sequentially using Lamport timestamps, so the first transaction in consensus order wins and subsequent commits see the target already filled.

### Submission Reveal

Within `REVEAL_WINDOW_MS` (~30,000ms) of commitment:

```rust
RevealSubmission {
    target_id: TargetId,
    data_hash: Hash,           // Hash of data (data hosted off-chain at data_url)
    data_url: String,          // Public URL for challenge verification
    embedding: Vector<f32>,    // Pre-computed embedding (verified only on challenge)
    nonce: Nonce,
}
```

**RevealSubmission Execution:**

1. Validate `hash(data_hash || embedding || nonce) == commitment`
2. Compute distance: `distance = ||embedding - target.embedding||` (simple L2 norm)
3. Validate `distance < target.radius`
4. Record as valid hit:
   - Add to `hits[current_epoch]`
   - Store submission details for challenge window

Note: Embedding is provided by the submitter and trusted at reveal time. Validators only re-compute the embedding during challenge audit to verify it matches what the model produces. The model used is implicitly the target's assigned model.

**Reveal Failure (Lazy Cleanup):**

Expired commitments are cleaned up lazily when the next miner attempts to commit to the same target. When a `CommitSubmission` targets a `Filled` target, execution first checks if the existing commitment has expired (no reveal and `current_timestamp_ms > commit_timestamp_ms + REVEAL_WINDOW_MS`). If expired:

- Previous miner's bond slashed to EmissionPool
- Target status reverts to `Open`
- New commit proceeds normally

This avoids needing separate timeout transactions or validator-driven cleanup. If nobody else wants the target, there's no urgency to reopen it.

### Submission Object

```rust
Submission {
    id: SubmissionId,
    target_id: TargetId,
    miner: Address,

    // Committed state
    commitment: Hash,  // hash(data_hash || embedding || nonce)
    commit_timestamp_ms: u64,
    bond_amount: u64,

    // Revealed state (set after successful reveal)
    data_hash: Option<Hash>,
    data_url: Option<String>,  // Public URL for challenge verification
    embedding: Option<Vector<f32>>,
    distance: Option<f32>,
    reveal_epoch: Option<u64>,
    reveal_timestamp_ms: Option<u64>,  // For difficulty adjustment
}
```

Note: `model_id` is not stored on submission since it's always the target's assigned model.

---

## Challenge System

### Challenge Window

Hits in epoch N can be challenged until the end of epoch N+1.

This extended window ensures that even hits occurring at the very end of an epoch have adequate time for verification.

### Challenge Types

**ScoreFraud**: Miner lied about distance

- Challenger downloads data from public URL
- Recomputes embedding and distance
- If `|computed - claimed| > epsilon`: fraud proven

**DataFraud**: Data doesn't match commitment

- Data at public URL doesn't hash to committed `data_hash`

### Challenge Object

```rust
Challenge {
    id: ChallengeId,
    submission_id: SubmissionId,
    challenger: Address,
    challenge_type: ChallengeType,
    challenger_bond: u64,
}

enum ChallengeType {
    ScoreFraud,
    DataFraud,
}
```

### InitiateChallenge Execution

1. Validate submission exists and is in challenge window (`reveal_epoch == current_epoch` or `current_epoch - 1`)
2. Validate no active challenge exists for this submission
3. Validate challenger bond >= `submission.bond_amount / 3`
4. Create Challenge object
5. Lock challenger bond
6. Validators begin audit

### Validator Audit

When a Challenge is created, each validator:

1. Download data from submission's public URL
2. Verify `hash(data) == data_hash` (DataFraud check)
3. Load the model
4. Compute embedding and distance
5. Compare against claimed values (ScoreFraud check)
6. Vote on verdict

### Challenge Resolution

Validators broadcast `ChallengeAuditVote` via P2P. When 2f+1 votes agree:

**If ChallengerWins:**

- Miner's bond transferred to challenger
- Submission removed from `hits[reveal_epoch]`
- Target reopens with same (E, R, models)
- Embedding NOT added to model index

**If MinerWins:**

- Challenger's bond transferred to miner
- Submission remains valid

### Challenge Timeout

If no 2f+1 quorum by end of challenge window:

- Challenge expires
- Challenger bond returned
- Submission remains valid (benefit of the doubt to miner)

---

## Reward Distribution

### Epoch Boundary Settlement

At the end of epoch N+1 (when challenge window for epoch N closes):

1. **Compute confirmed hits:**
   ```
   confirmed_hits[N] = hits[N] - successfully_challenged_submissions
   ```

2. **Distribute rewards for each confirmed hit:**
   ```
   per_hit_reward = epoch_N_target_emissions / num_confirmed_hits
   miner_share  = per_hit_reward × 50%
   model_share  = per_hit_reward × 30%
   ```

   If no confirmed hits: return target emissions to EmissionPool

3. **Update model statistics:**
   - Increment `model.total_rewards_earned`
   - Confirm embedding addition to model index

4. **Return bonds** to confirmed hit miners

### Validator Reward Distribution

Validators receive 20% of total epoch emissions, distributed proportionally by stake pool (same as current emissions logic). The remaining 80% goes to target winners (miners and models).

---

## Difficulty Adjustment

### Mechanism

Difficulty controls the target radius R, which implicitly controls:

- How hard it is to find valid data
- How many targets are concurrently open
- Quality threshold (tighter radius → better data needed)

### Adjustment Algorithm

At each epoch boundary:

```
avg_hit_time = mean(hit_times for confirmed_hits[N])

if avg_hit_time < target_hit_time:
    radius = radius × (1 - adjustment_rate)  # Harder
elif avg_hit_time > target_hit_time:
    radius = radius × (1 + adjustment_rate)  # Easier

radius = clamp(radius, min_radius, max_radius)
```

### Bounds

- **min_radius**: Prevents impossibly hard targets; ensures reasonable hit rate
- **max_radius**: Ensures minimum quality threshold even when data is sparse

### Per-Region Adjustment (Future)

Different embedding regions may have varying data density. Future versions may track per-region difficulty to ensure coverage across the full space.

---

## Timing Summary

### Continuous Operations

- Target generation (on hit)
- Submission commits and reveals
- Mining competition
- Challenge initiation

### Epoch-Boundary Operations

- Model commits revealed
- Model index updated (new models + confirmed embeddings from epoch N-1)
- Challenge windows close for epoch N-1
- Reward distribution for epoch N-1
- Difficulty recalculation

---

## Step-Decay Emissions

### EmissionPool State

```rust
EmissionPool {
    total_supply: u64,
    distributed: u64,
    distribution_counter: u64,
    current_epoch_emissions: u64,
    subsidy_period_length: u64,      // e.g., 365 epochs
    subsidy_decrease_rate_bps: u64,  // e.g., 1000 = 10%
}
```

### Decay Schedule

At each epoch boundary:

```
distribution_counter += 1

if distribution_counter % subsidy_period_length == 0:
    current_epoch_emissions = current_epoch_emissions × (10000 - subsidy_decrease_rate_bps) / 10000
```

### Allocation Split

Each epoch:
- 20% to validators (proportional to stake)
- 80% to target winner pool (split 50/30 between miners/models)

---

## Data Availability

### Public Data Corpus

All submitted data must be publicly available for the challenge window duration.

Miners must:

- Host data at a publicly accessible URL
- Include URL in reveal transaction
- Maintain availability through end of challenge window (epoch N+1)

Failure to maintain availability during an active challenge results in automatic challenge success (MinerUnresponsive verdict).

---

## Configuration Parameters

```rust
// Model registration
model_min_stake: u64                    // Minimum stake to register a model
model_architecture_version: u64         // Current supported architecture version
model_reveal_slash_rate_bps: u64        // Slash rate for failed reveals (e.g., 5000 = 50%)
model_tally_threshold: u64              // Number of tallies before removal
model_tally_slash_rate_bps: u64         // Slash rate when removed due to tallies
embedding_dim: u64                      // Embedding dimension

// Target generation
target_generation_budget_per_epoch: u64

// Submissions
submission_min_bond: u64
reveal_window_ms: u64  // ~30,000ms (30 seconds)
distance_epsilon: f32  // Tolerance for floating-point comparisons

// Difficulty
target_hit_time: u64  // target average time to fill
adjustment_rate: f32
min_radius: f32
max_radius: f32

// Challenges
challenge_bond_ratio: f32  // 1/3 of miner bond

// Rewards
miner_reward_share_bps: u64   // 5000 = 50%
model_reward_share_bps: u64   // 3000 = 30%
validator_reward_share_bps: u64  // 2000 = 20%

// Emissions
subsidy_period_length: u64
subsidy_decrease_rate_bps: u64
```

---

## Migration Notes

### Removed from Previous Design

- Wave pipeline (3-epoch overlapping structure)
- BetterModel challenges (models selected by system, not miner)
- Per-submission tBLS reveals (replaced with hash-commit-reveal)
- Rankings and winner bumping (replaced with target reopening)
- Private buyer targets (deferred to future version)
- Reconstruction score as win criteria (radius-based only)
- Encoder committee and shard-based processing
- VDF-based randomness for shard selection

### Removed from Initial Redesign (Simplified)

- Threshold BLS for model reveals (replaced with simple commit-reveal)
- kNN index for model selection (replaced with stake-weighted sampling)
- Multiple models per target (simplified to single model)
- Model capability embeddings (not needed without kNN)

### New Concepts

- Model staking with delegation (stake as reputation proxy)
- Simple commit-reveal for model registration (commit in N, reveal in N+1)
- Stake-weighted model selection for targets
- Model availability audits with tally-based removal
- Race-based winner selection (first valid commit)
- Extended challenge windows (epoch N hits challengeable through epoch N+1)
- Difficulty-controlled radius
- 50/30/20 reward split (miner/model/validator)
