# Challenge → Validator Audit Migration Plan

## Status: Stage 1 Complete

Stage 1 has been fully implemented. All challenge types, transactions, execution handlers, CLI commands, RPC endpoints, and tests have been removed. The audit service is temporarily defunct. All workspace tests pass.

---

## Problem

The challenge epoch was set to `fill_epoch` (same epoch the submission lands in). Validators need the epoch to close first, then the *next* epoch is the audit window. This keeps the 2f+1 stake calculation for reports contained to a single, known validator set.

Rather than fix the epoch timing on a system being removed, we removed challenges entirely in Stage 1. The audit service is temporarily defunct (no trigger mechanism). Stage 2 brings it back as an autonomous validator-driven flow.

---

## Epoch Timeline (After Stage 1)

```
Epoch E:     Target filled (fill_epoch = E)

Epoch E+1:   Audit window — validators can report via ReportSubmission
             ReportSubmission allowed (epoch == fill_epoch + 1 ONLY)
             UndoReportSubmission allowed (epoch == fill_epoch + 1 ONLY)

Epoch E+2:   ClaimRewards allowed (epoch > fill_epoch + 1)
             Two outcomes:
               - 2f+1 report quorum → fraud, bond to reporting validators
               - No quorum → valid, rewards distributed, bond returned
```

---

## Stage 1: Remove Challenges — COMPLETED

### What Was Removed

**Deleted files:**
- `types/src/challenge.rs` — `ChallengeV1`, `ChallengeStatus`, all challenge types
- `authority/src/execution/challenge.rs` — all challenge execution handlers
- `cli/src/commands/challenge.rs` — CLI challenge commands
- `e2e-tests/tests/challenge_tests.rs` — challenge e2e tests
- `rpc/src/api/grpc/state_service/get_challenge.rs` — gRPC handler
- `rpc/src/api/grpc/state_service/list_challenges.rs` — gRPC handler

**Transaction types removed:**
- `InitiateChallenge`, `ReportChallenge`, `UndoReportChallenge`, `ClaimChallengeBond` — all removed from `TransactionKind` enum in `types/src/transaction.rs`
- `InitiateChallengeArgs` removed
- `is_challenge_tx()` helper removed

**Object type removed:**
- `ObjectType::Challenge` removed from `types/src/object.rs`
- `CHALLENGE_OBJECT_SHARED_VERSION` removed from `types/src/lib.rs`
- All challenge object creation/loading/saving helpers removed

**Target fields removed:**
- `challenger: Option<SomaAddress>` removed from `TargetV1`
- `challenge_id: Option<ChallengeId>` removed from `TargetV1`

**Type changes:**
- `TargetV1.submission_reports`: `BTreeMap<SomaAddress, Option<SomaAddress>>` → `BTreeSet<SomaAddress>`
- `report_submission(signer, challenger)` → `report_submission(signer)`
- `get_submission_report_quorum()` returns `(bool, Vec<SomaAddress>)` instead of `(bool, Option<SomaAddress>, Vec<SomaAddress>)`
- `ReportSubmission { target_id, challenger }` → `ReportSubmission { target_id }`

**Execution changes:**
- `ReportSubmission` window tightened: `epoch == fill_epoch + 1` only (was `epoch <= fill_epoch + 1`)
- `UndoReportSubmission` window tightened to same constraint
- `ClaimRewards` simplified to two outcomes (was three — removed challenger payout path)

**Error renames:**
- `ChallengeWindowOpen` → `AuditWindowOpen`
- `ChallengeWindowClosed` + 7 other challenge errors → `AuditWindowClosed`

**Infrastructure removed:**
- Challenge observation pipeline from checkpoint executor
- Audit service `spawn()` method (core audit logic retained)
- `challenger_bond_per_byte` from `ProtocolConfig` and `SystemParameters`
- Challenge RPC index (`ChallengeIndexKey`, `ChallengeIndexInfo`, `challenges_iter`)
- Challenge methods from RPC client, SDK, Python SDK

**Proto handling:**
- Proto-generated challenge variants handled with error fallback in `transaction.rs`
- `challenger_bond_per_byte` set to `None` in proto conversions
- `.proto` files NOT edited (still contain challenge service definitions — can be cleaned up separately)

### What Was Kept (for Stage 2 reuse)

- **`audit_service.rs`** — `audit_fraud(&TargetV1)` and `manifest_competition()` logic retained. The service can't trigger without a spawn mechanism, but the core audit/scoring logic is intact.
- **`ReportSubmission` / `UndoReportSubmission`** — execution handlers fully functional with new epoch constraints.
- **`ClaimRewards`** — handles quorum-based fraud detection with validator bond distribution.
- **Target RPC index** — `TargetIndexKey { status, epoch, id }` table in `rpc_index.rs` supports querying filled targets by epoch.
- **Submission report tally** — `BTreeSet<SomaAddress>` on `TargetV1`, `get_submission_report_quorum()` using validator voting power.

---

## Stage 2: Autonomous Validator Auditing (Post-Testnet)

No protocol changes needed. All on-chain primitives (`ReportSubmission`, `ClaimRewards`, tally quorum) already work after Stage 1. Stage 2 is pure validator-side infrastructure.

### 2.1 Validator-Side Filled Target Tracker

**Build a `ValidatorTargetTracker` component that:**
- Listens to committed checkpoints (same hook point where `observe_created_challenges` used to be in `checkpoint_executor/mod.rs`)
- Tracks targets that transitioned to `Filled` during the current epoch
- On epoch boundary, snapshots the set as the audit work queue for the next epoch
- Stores in a simple in-memory set keyed by `(fill_epoch, target_id)`
- Discards data after the audit window (E+1) closes

The existing `TargetIndexKey` table in `rpc_index.rs` can also be queried as a fallback.

### 2.2 Probabilistic Sampling Strategy

At the start of each epoch E+1, each validator:

1. Queries the filled target tracker for all targets filled in epoch E
2. Determines its sampling set using a deterministic seed (e.g., `hash(validator_address, epoch)`)
3. Sampling rate guarantees each submission is checked by at least one validator with high probability:
   - Each validator samples `ceil(M * k / N)` targets where `k` is an overcoverage factor (e.g., 3x)
   - With k=3 and N=4 validators, each validator checks ~75% of targets
   - Probability of a target being unchecked = `(1 - k/N)^N` → negligible for reasonable k

### 2.3 Audit Service Refactor

**Wire the existing audit logic to the new trigger:**

1. **Trigger:** Epoch change event (not challenge creation). Subscribe to epoch store events.
2. **Sample:** Apply probabilistic sampling from 2.2 to select targets.
3. **Audit:** For each sampled target, call `audit_fraud()` (already in `audit_service.rs` — calls scoring runtime, compares results).
4. **Report:** If fraud found, submit `ReportSubmission` tx. No challenge object needed.
5. **No report if valid:** Validators do nothing for valid submissions. Absence of 2f+1 reports = submission is implicitly valid.

**Key changes to `audit_service.rs`:**
- Re-add `spawn()` with new trigger: epoch-change listener
- Add filled target query interface (via `ValidatorTargetTracker` or RPC index)
- Keep `audit_fraud()` and report submission logic as-is

### 2.4 Node Configuration

Validators need:
- Scoring runtime endpoint (already exists in config)
- Audit sampling parameters (overcoverage factor, optional manual sample rate override)
- Enable/disable flag for the audit service (for gradual rollout)

---

## Summary

| | Stage 1 (Pre-Testnet) | Stage 2 (Post-Testnet) |
|---|---|---|
| **Status** | **COMPLETED** | Not started |
| **Scope** | Remove challenges from protocol | Add autonomous validator auditing |
| **Protocol upgrade** | Yes (ships with testnet genesis) | No |
| **Challenge txs** | Removed | N/A |
| **Challenge objects** | Removed | N/A |
| **Audit service** | Defunct (no trigger) | Active (epoch-triggered) |
| **Fraud detection** | None (acceptable for testnet) | Validators autonomously sample + verify |
| **Report flow** | `ReportSubmission` exists but no one calls it | Validators call `ReportSubmission` directly |
| **Bond model** | Submitter bond only | Submitter bond only |
| **Reward claims** | 2 outcomes (quorum → fraud, else → valid) | Same |
