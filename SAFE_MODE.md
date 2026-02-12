# Safe Mode: Liveness Guarantee for Epoch Transitions

Research document covering Sui's safe mode infrastructure, Soma's current gap, and an actionable implementation plan.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [How Sui Solves It](#how-sui-solves-it)
3. [Soma's Current Exposure](#somas-current-exposure)
4. [Proposed Implementation](#proposed-implementation)
5. [Detection and Alerting](#detection-and-alerting)
6. [Practical Example: Model Architecture Upgrade Gone Wrong](#practical-example-model-architecture-upgrade-gone-wrong)
7. [Cost-Benefit Analysis](#cost-benefit-analysis)
8. [Implementation Checklist](#implementation-checklist)

---

## The Problem

Every epoch transition executes a `ChangeEpoch` transaction that runs complex processing: validator reward distribution, model registry updates (slashing, reveal timeouts, staking pool processing), difficulty adjustment, target generation, and protocol version upgrades. **If any of this fails, the chain halts permanently.**

Today in Soma, the failure path is even worse than a halt — it's a crash. In `authority.rs:2286`:

```rust
// The change epoch transaction cannot fail to execute.
assert!(effects.status().is_ok());
```

If `ChangeEpochExecutor::execute()` returns any error, the effects have a failure status, this assert panics, and the validator crashes. Since all validators execute the same `ChangeEpoch` transaction deterministically, **every validator crashes simultaneously**. The network is dead with no automated recovery path.

The risk compounds with each new phase of development. Today's epoch boundary runs:
- Validator reward calculation + slashing (~80 lines)
- Model registry processing: report quorum, reveal timeouts, update cancellation, commission adjustments, staking pools (~120 lines)
- Target state: difficulty adjustment, reward calculation (~90 lines)
- Target object generation loop (~60 lines)

Future phases add submission reward distribution, challenge window closure, and bond settlement — each adding more failure surface area to the single most critical transaction in the system.

---

## How Sui Solves It

Sui's safe mode is a **liveness guarantee**: if `ChangeEpoch` execution fails, the network still transitions to a new epoch by performing a minimal bump.

### SystemState Fields

Sui adds 5 fields to their system state:

```rust
pub struct SuiSystemStateInnerV2 {
    // ... normal fields ...

    /// Whether the system is running in a downgraded safe mode due to a non-recoverable bug.
    /// This is set whenever we failed to execute advance_epoch, and ended up executing
    /// advance_epoch_safe_mode. It can be reset once we are able to successfully execute
    /// advance_epoch.
    safe_mode: bool,

    /// Amount of storage rewards accumulated (and target stake subsidies) during safe mode.
    safe_mode_storage_rewards: Balance,
    /// Amount of computation rewards accumulated during safe mode.
    safe_mode_computation_rewards: Balance,
    /// Amount of storage rebates accumulated during safe mode.
    safe_mode_storage_rebates: u64,
    /// Amount of non-refundable storage fee accumulated during safe mode.
    safe_mode_non_refundable_storage_fee: u64,
}
```

The holding fields accumulate fees/rewards that would normally be distributed during the epoch transition. They're drained on the next successful epoch transition.

### Try-Catch Pattern in the Executor

In Sui's execution engine (`execution_engine.rs`), the flow is:

```
1. Attempt normal advance_epoch()
2. If it succeeds → normal path, effects committed
3. If it fails:
   a. Drop ALL writes from the failed attempt
   b. Reset gas meter
   c. Execute advance_epoch_safe_mode() instead
   d. This minimal path always succeeds
   e. Commit the safe mode effects
```

The critical insight: the safe mode path is so simple it **cannot fail**. It only does:
- Set `safe_mode = true`
- Increment `epoch`
- Accumulate fees into the 4 holding fields
- Skip all complex processing (rewards, staking, slashing, etc.)

### Recovery Path

On the next epoch transition, `advance_epoch()` checks the `safe_mode` flag:
- If `true`: drain all holding fields into the normal reward pools, set `safe_mode = false`, then proceed with normal processing
- The bug that caused safe mode entry can be fixed via protocol upgrade between epochs

### Testing

Sui's `safe_mode_reconfig_test` uses a failpoint `advance_epoch_result_injection` to inject failures for specific epoch ranges:

```rust
// Fail epochs 2 and 3, recover at epoch 4
advance_epoch_result_injection::set_override(Some((2, 3)));
```

The test verifies:
1. Normal transactions still work during safe mode epochs
2. The network recovers automatically when the failure is removed
3. Accumulated rewards are properly distributed after recovery

---

## Soma's Current Exposure

### Failure Points in `advance_epoch()`

Every `?` operator and every arithmetic operation is a potential chain-killer. Here's the full inventory:

#### Critical: Arithmetic Overflow

| Location | Code | Risk |
|----------|------|------|
| `mod.rs:1048` | `total_rewards += self.emission_pool.advance_epoch()` | u64 overflow (unchecked `+=`) |
| `mod.rs:1064` | `(total_rewards * validator_allocation_bps) / BPS_DENOMINATOR` | Multiplication overflow |
| `mod.rs:1089` | `self.emission_pool.balance += validator_reward_pool` | u64 overflow |
| `mod.rs:1107` | `(epoch_emissions * target_allocation_bps) / BPS_DENOMINATOR` | **Overflow**: `emission_per_epoch` can be ~10^18, `target_allocation_bps` up to 10000 → product exceeds u64 max (1.8×10^19) |
| `mod.rs:1011` | `.map(|m| m.staking_pool.soma_balance).sum()` | Overflow if many large staking pools |

#### Critical: Logic Bugs Amplified by No Recovery

| Location | Code | Risk |
|----------|------|------|
| `mod.rs:930` | `saturating_sub(model.staking_pool.soma_balance + slash_amount)` | Double-subtraction: `soma_balance` already reduced by `slash_amount` on line 925, then subtracts `soma_balance + slash_amount` from total — overcounts by `slash_amount` |
| `mod.rs:898` | `let prev_epoch = new_epoch - 1` | Underflow if `new_epoch == 0` (should never happen, but no guard) |

#### Medium: Panicking Operations

| Location | Code | Risk |
|----------|------|------|
| `change_epoch.rs:45-51` | `bcs::from_bytes::<SystemState>()` | BCS deserialization failure crashes |
| `change_epoch.rs:149-154` | `bcs::to_bytes(&state)` | BCS serialization failure crashes |
| `change_epoch.rs:109-116` | `generate_target()` | Model selection can fail (currently handled with `break`) |
| `mod.rs:1155-1156` | `f32` arithmetic for difficulty | NaN/Inf from degenerate parameters |

#### The Fatal Assert

```rust
// authority.rs:2268-2286
let (transaction_outputs, _execution_error_opt) = self
    .execute_certificate(&execution_guard, &executable_tx, input_objects, None, epoch_store)
    .unwrap();  // ← unwrap on execute_certificate result

// ...

// The change epoch transaction cannot fail to execute.
assert!(effects.status().is_ok());  // ← panics ALL validators simultaneously
```

If any failure point above triggers, ALL validators crash in lockstep.

---

## Proposed Implementation

### Design Principles

Adapted from Sui's approach but simplified for Soma's architecture (no MoveVM, no storage rebates):

1. **Safe mode does the absolute minimum**: increment epoch, accumulate fees, skip everything else
2. **Safe mode path cannot fail**: no complex math, no loops, no external calls
3. **Recovery is automatic**: next successful epoch drains holding fields
4. **Detection is immediate**: metrics + logging on safe mode entry

### New SystemState Fields

```rust
pub struct SystemState {
    // ... existing fields ...

    /// Whether the system is in safe mode due to a failed epoch transition.
    /// Set to true when advance_epoch() fails; reset to false on next successful advance.
    pub safe_mode: bool,

    /// Transaction fees accumulated during safe mode epochs, waiting to be distributed.
    pub safe_mode_accumulated_fees: u64,

    /// Emission rewards accumulated during safe mode epochs, waiting to be distributed.
    pub safe_mode_accumulated_emissions: u64,
}
```

Only 3 new fields (vs Sui's 5) because Soma has no storage rebate system. The two holding fields (`accumulated_fees` and `accumulated_emissions`) are sufficient to capture everything that would normally be processed.

### Modified ChangeEpochExecutor

```rust
// In ChangeEpochExecutor::execute():

// Clone state before attempting advance_epoch
let state_backup = state.clone();

match state.advance_epoch(/* ... */) {
    Ok(validator_rewards) => {
        // Normal path: create reward objects, generate targets, etc.
        // (existing code, unchanged)
    }
    Err(e) => {
        // Safe mode: restore state from backup, do minimal epoch bump
        error!("advance_epoch FAILED, entering safe mode: {:?}", e);

        state = state_backup;
        state.advance_epoch_safe_mode(
            change_epoch.epoch,
            change_epoch.fees,
            epoch_start_timestamp_ms,
        );
    }
}

// Serialize and commit (this path is shared — always succeeds)
```

### New `advance_epoch_safe_mode()` Method

```rust
impl SystemState {
    /// Minimal epoch transition when normal advance_epoch fails.
    /// Cannot fail — only increments epoch and accumulates fees.
    pub fn advance_epoch_safe_mode(
        &mut self,
        new_epoch: u64,
        epoch_fees: u64,
        epoch_start_timestamp_ms: u64,
    ) {
        self.safe_mode = true;
        self.epoch = new_epoch;
        self.epoch_start_timestamp_ms = epoch_start_timestamp_ms;

        // Accumulate fees — will be distributed on recovery
        self.safe_mode_accumulated_fees =
            self.safe_mode_accumulated_fees.saturating_add(epoch_fees);

        // Accumulate emissions if the epoch duration elapsed
        let emission = std::cmp::min(
            self.emission_pool.emission_per_epoch,
            self.emission_pool.balance,
        );
        self.emission_pool.balance = self.emission_pool.balance.saturating_sub(emission);
        self.safe_mode_accumulated_emissions =
            self.safe_mode_accumulated_emissions.saturating_add(emission);

        // No validator rewards, no model processing, no target generation,
        // no difficulty adjustment, no staking pool processing.
        // The committee, parameters, and all registries remain frozen.
    }
}
```

### Recovery in `advance_epoch()`

At the top of the existing `advance_epoch()`, add:

```rust
// Drain safe mode accumulators if recovering from safe mode
if self.safe_mode {
    info!("Recovering from safe mode — draining accumulated rewards");
    total_rewards += self.safe_mode_accumulated_fees;
    total_rewards += self.safe_mode_accumulated_emissions;
    self.safe_mode_accumulated_fees = 0;
    self.safe_mode_accumulated_emissions = 0;
    self.safe_mode = false;
}
```

The accumulated fees and emissions flow into the normal reward distribution path, so validators and the emission pool get their correct share — just delayed by one epoch.

### Remove the Fatal Assert

Replace `authority.rs:2285-2286`:

```rust
// Before:
assert!(effects.status().is_ok());

// After:
if !effects.status().is_ok() {
    error!(
        "ChangeEpoch transaction failed with status: {:?}. \
         This should not happen — safe mode should have caught this.",
        effects.status()
    );
}
```

With safe mode, the assert should never fire (safe mode always succeeds). But converting it to a logged error provides defense-in-depth rather than crashing the network on an unexpected edge case.

---

## Detection and Alerting

### How Safe Mode Entry Is Detected

**1. Logs (immediate)**

The `error!()` log in the executor is the first signal:

```
ERROR authority::execution::change_epoch: advance_epoch FAILED, entering safe mode: <error details>
```

This fires on every validator simultaneously and includes the exact failure reason.

**2. SystemState query (programmatic)**

Any RPC client can detect safe mode:

```rust
let state = client.get_system_state().await?;
if state.safe_mode {
    alert!("NETWORK IN SAFE MODE since epoch {}", state.epoch);
}
```

**3. Prometheus metrics (monitoring)**

Add a gauge that operators can alert on:

```rust
// In epoch metrics
pub static ref IS_SAFE_MODE: IntGauge = register_int_gauge!(
    "soma_is_safe_mode",
    "Whether the network is in safe mode (0 = normal, 1 = safe mode)"
).unwrap();
```

Updated at every epoch transition. Any monitoring stack (Grafana, Datadog, PagerDuty) can fire alerts on `soma_is_safe_mode == 1`.

**4. CLI**

```bash
soma system-state | grep safe_mode
# safe_mode: true
# safe_mode_accumulated_fees: 1500000000
# safe_mode_accumulated_emissions: 50000000000000
```

### What Operators See

When safe mode activates:
- Validators keep running and producing blocks
- User transactions (transfers, staking, submissions) continue normally
- Epoch transitions continue (just minimal)
- No validator rewards are distributed (accumulated instead)
- No model registry processing (models frozen in current state)
- No new targets are generated
- No difficulty adjustment occurs

The network is **degraded but alive**. This is infinitely better than dead.

---

## Practical Example: Model Architecture Upgrade Gone Wrong

### Scenario

You ship protocol version 3 which introduces `architecture_version: u64` to the `Model` struct for a new probe architecture. The genesis models have `architecture_version = 1`. The upgrade code in `advance_epoch_models()` is supposed to migrate existing models, but there's a bug:

```rust
// The buggy migration code in advance_epoch_models():
for model in self.model_registry.active_models.values_mut() {
    // Bug: divides by (architecture_version - 1), which is 0 for v1 models
    let scaling_factor = model.embedding_dim / (model.architecture_version - 1);
    model.embedding_dim = scaling_factor * 2;
}
```

### Without Safe Mode

1. **Epoch N ends.** Validators create the `ChangeEpoch` transaction.
2. `advance_epoch()` → `advance_epoch_models()` → division by zero on the first v1 model.
3. `ChangeEpochExecutor::execute()` returns `Err(ExecutionFailureStatus::SomaError(...))`.
4. Effects have failure status.
5. `assert!(effects.status().is_ok())` fires on **every validator**.
6. **All validators crash simultaneously.**
7. Network is dead. No blocks. No transactions. No recovery path.

**Resolution requires manual intervention:**
- Operators must coordinate out-of-band (Discord, Telegram)
- Someone must build a patched binary that either fixes the bug or hard-forks the chain
- All validators must upgrade and restart
- If the bug corrupted any on-disk state, a genesis reset may be required
- **Downtime: hours to days**

### With Safe Mode

1. **Epoch N ends.** Validators create the `ChangeEpoch` transaction.
2. `advance_epoch()` → `advance_epoch_models()` → division by zero.
3. `advance_epoch()` returns `Err`.
4. Executor catches the error, restores `state_backup`, calls `advance_epoch_safe_mode()`.
5. Safe mode succeeds: epoch incremented, fees accumulated, `safe_mode = true`.
6. Effects have **success** status. Assert passes. Network continues.
7. **Alert fires**: `soma_is_safe_mode == 1`, error logs on every validator.

**Resolution:**
- Operators see the alert and the exact error in logs: `"division by zero in advance_epoch_models"`
- Team ships a fix in protocol version 4:
  ```rust
  let scaling_factor = model.embedding_dim / model.architecture_version.max(1);
  ```
- Validators upgrade to the patched binary during the current (safe mode) epoch
- At the next epoch boundary, `advance_epoch()` succeeds:
  - Drains `safe_mode_accumulated_fees` and `safe_mode_accumulated_emissions` into normal reward distribution
  - Sets `safe_mode = false`
  - Runs all deferred processing (model migration now works correctly)
- **Downtime: zero.** Degraded operation for 1-2 epochs while the fix is deployed.

### What "Degraded" Means in Practice

During the safe mode epoch(s):
- Users can still send SOMA, stake, transfer objects — all normal transactions work
- No validator rewards are minted (accumulated for later)
- Model registry is frozen (no new models activate, no reports processed, no slashing)
- No new targets are generated (existing targets remain active)
- Difficulty doesn't adjust
- When safe mode exits, all accumulated rewards are distributed and deferred processing runs

---

## Cost-Benefit Analysis

### Implementation Cost

| Component | Files Modified | Lines Changed | Complexity |
|-----------|---------------|---------------|------------|
| SystemState fields | `types/src/system_state/mod.rs` | ~10 | Trivial (3 new fields) |
| `advance_epoch_safe_mode()` | `types/src/system_state/mod.rs` | ~25 | Trivial (no complex logic) |
| Recovery drain in `advance_epoch()` | `types/src/system_state/mod.rs` | ~15 | Trivial (add to existing rewards) |
| Try-catch in executor | `authority/src/execution/change_epoch.rs` | ~20 | Low (clone + match) |
| Remove fatal assert | `authority/src/authority.rs` | ~5 | Trivial |
| Genesis default | `types/src/genesis_builder.rs` | ~3 | Trivial |
| Metric | `authority/src/epoch_metrics.rs` (or similar) | ~10 | Trivial |
| Failpoint + E2E test | `e2e-tests/tests/failpoint_tests.rs` | ~60 | Low |
| **Total** | **~5 files** | **~150 lines** | **1-2 days** |

### Benefit

| Scenario | Without Safe Mode | With Safe Mode |
|----------|-------------------|----------------|
| Arithmetic overflow in reward calc | **All validators crash. Chain dead.** | 0 downtime. Fix deployed during degraded epoch. |
| Bug in model migration during protocol upgrade | **All validators crash. Chain dead.** | 0 downtime. Rollback or fix in next protocol version. |
| Unexpected edge case in difficulty adjustment (NaN, Inf) | **All validators crash. Chain dead.** | 0 downtime. Parameters adjusted. |
| BCS serialization issue from state corruption | **All validators crash. Chain dead.** | 0 downtime. State frozen at last good snapshot. |
| Division by zero in new phase code (submissions, challenges) | **All validators crash. Chain dead.** | 0 downtime. Fix shipped within the epoch. |

### Risk Calculus

- **Probability of epoch boundary bug on testnet**: High. This is new code being exercised with real-world patterns for the first time. Every new phase (targets, submissions, challenges, rewards) adds ~50-100 lines of epoch boundary logic.
- **Impact without safe mode**: Total network outage requiring coordinated manual intervention.
- **Impact with safe mode**: Degraded operation (no rewards/targets) for 1-2 epochs while a fix is deployed.
- **Implementation cost**: ~150 lines, 1-2 days.
- **Maintenance cost**: Near zero. The safe mode path is ~25 lines that never change (it deliberately does nothing complex).

The expected value is overwhelmingly positive. Safe mode turns a catastrophic, reputation-destroying outage into a manageable operational event.

---

## Implementation Checklist

### Phase 1: Core Infrastructure (~100 lines)

- [ ] **Add 3 fields to `SystemState`** (`types/src/system_state/mod.rs`)
  - `safe_mode: bool` (default `false`)
  - `safe_mode_accumulated_fees: u64` (default `0`)
  - `safe_mode_accumulated_emissions: u64` (default `0`)

- [ ] **Add `advance_epoch_safe_mode()` method** (`types/src/system_state/mod.rs`)
  - Set `safe_mode = true`
  - Increment epoch + timestamp
  - Accumulate fees and emissions into holding fields
  - No other processing

- [ ] **Add recovery drain to `advance_epoch()`** (`types/src/system_state/mod.rs`)
  - At top of method: if `self.safe_mode`, drain accumulators into `total_rewards`, set `safe_mode = false`

- [ ] **Wrap executor in try-catch** (`authority/src/execution/change_epoch.rs`)
  - Clone `state` before `advance_epoch()`
  - On `Err`: restore from clone, call `advance_epoch_safe_mode()`, log error
  - Ensure target generation is also inside the try-catch (it's after `advance_epoch()` today)

- [ ] **Replace fatal assert** (`authority/src/authority.rs:2286`)
  - Convert `assert!(effects.status().is_ok())` to an `error!()` log

- [ ] **Update genesis** (`types/src/genesis_builder.rs`)
  - Initialize new fields to defaults (`false`, `0`, `0`)

### Phase 2: Detection (~30 lines)

- [ ] **Add Prometheus metric** for `is_safe_mode` gauge
  - Update on every epoch transition
  - Operators can alert on `soma_is_safe_mode == 1`

- [ ] **Add `safe_mode` to SystemState RPC response**
  - Ensure `get_system_state` / `get_epoch` returns the safe mode fields

### Phase 3: Testing (~60 lines)

- [ ] **Add failpoint tag** `advance_epoch_result_injection` in `advance_epoch()` (`types/src/system_state/mod.rs`)
  - Under `#[cfg(msim)]`: check thread-local override, return injected error for specified epochs

- [ ] **Add E2E test** `test_safe_mode_reconfig` in `failpoint_tests.rs`
  - Inject failure for epoch 2
  - Verify safe mode activates (check `system_state.safe_mode == true`)
  - Verify transactions still work during safe mode
  - Verify recovery at epoch 3 (safe mode deactivates, accumulated rewards distributed)

- [ ] **Add E2E test** `test_safe_mode_multi_epoch` in `failpoint_tests.rs`
  - Inject failure for epochs 2-3
  - Verify fees accumulate across multiple safe mode epochs
  - Verify single recovery drains all accumulated rewards

### Phase 4: Hardening (optional, post-testnet)

- [ ] Fix arithmetic overflow in `calculate_reward_per_target` (use u128 intermediate)
- [ ] Fix double-subtraction bug in model slashing (`mod.rs:930`)
- [ ] Replace unchecked `+=` with `saturating_add` in reward accumulation
- [ ] Add `checked_sum()` for `total_model_stake` computation
