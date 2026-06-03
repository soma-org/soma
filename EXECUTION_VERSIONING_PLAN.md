# Execution Versioning Plan

Goal: replay determinism across binary upgrades, and elimination of the
"ungated effect change → fork" class that has already caused a testnet fork.
Adapts Sui's `sui-execution` frozen-crate pattern to Soma's **native-Rust**
execution (no Move VM, so no verifier / move-natives / layout-resolver surface).

Status: pre-mainnet, with a regenesis pipeline. This is the key enabler — the
`latest` cut can be mutated freely until mainnet genesis; the first frozen cut
(`v0`) is taken **at** mainnet genesis.

## Current state (verified against the code)

- Dispatch + effect orchestration: `authority/src/execution/mod.rs`
  (`execute_transaction:110`, `create_executor:424`, `handle_shared_object_transaction:545`).
- Per-tx-kind executors: `authority/src/execution/{validator,change_epoch,settlement,channel,bridge,staking,balance_transfer,provider,offering,object,system}.rs`.
- **The heavy effect-determining logic lives in `types/`, not `authority/`** — this
  is the structural violation that makes the frozen-cut pattern not yet viable:
  - `types/src/temporary_store.rs`: `into_effects:581`, `get_object_changes:525`,
    accumulator-write folding `552-575`, `ensure_active_inputs_mutated`,
    `check_ownership_invariants`, lamport/version assignment, `emit_*` recorders.
  - `types/src/system_state/mod.rs`: `advance_epoch:499`, `advance_epoch_safe_mode:606`,
    stake/validator request family.
  - `types/src/system_state/{validator,staking,emission}.rs`: `ValidatorSet::advance_epoch`,
    reward/slashing/transitions, F1 `auto_settle`, `EmissionPool::advance_epoch`.
  - `types/src/accumulator.rs`: `BalanceAccumulator::derive_id:95`,
    `DelegationAccumulator::derive_id:150` (consensus-critical IDs, called from
    `execution/mod.rs:186,195,236`).
  - `types/src/{channel,provider,offering}.rs`: mutating/pricing methods.
- Versioning today: `execution_version: Option<u64>` in `protocol-config/src/lib.rs:224`
  (0@PV1, 1@PV2, 2@PV3). Only **5 inline `if execution_version >= 1`** gates exist,
  all in `execution/mod.rs` (288,319,406,569,609). Every other effect change in
  `types/` is ungated.

## Key decision: split `types`, don't move all of it

Recommend **(b) split `types` into a behavior-free shared part and an
effect-bearing versioned part**. Data definitions + serde + pure getters stay in
`types` (indexer/rpc/sdk/cli/node must keep deserializing them); **mutating /
effect-computing methods move into `soma-execution`.** Moving all of `types`
behind the boundary is wrong — it would drag wire-format structs into the
versioned crate.

## Target architecture (Soma-specific, no Move)

```
soma-execution/            # dispatcher + trait
  src/executor.rs          # trait Executor { execute_transaction(...) -> (InnerTemporaryStore, TransactionEffects, Option<ExecutionError>) }
  src/lib.rs               # executor(execution_version) -> Box<dyn Executor>; 0 => v0, _ => latest
  tests/encapsulation.rs   # authority/** may only reference soma_execution::, never a concrete vN
soma-execution-latest/     # the ONLY mutable cut (executors + carved-out types/ logic)
soma-execution-v0/         # frozen at mainnet genesis (Phase 5; byte-copy of latest)
```

`authority/src/execution/mod.rs` shrinks to a shim calling
`soma_execution::executor(ev).execute_transaction(...)`. The per-kind
`TransactionExecutor` trait stays internal to each cut.

## Phases (safest/cheapest → riskiest)

### Phase 1 — CI effect-surface guard (no effect change; ship now)
Add a `.github/workflows/ci.yml` job (mirror the existing `diff`/paths-filter)
that fails any change touching the effect surface
(`authority/src/execution/**`, `types/src/temporary_store.rs`,
`types/src/effects/**`, `types/src/system_state/**`, `types/src/accumulator.rs`,
`types/src/{channel,provider,offering}.rs`) **unless** the same diff bumps
`protocol-config/src/lib.rs` or carries an explicit `EFFECT-CHANGE-OK:`
justification. Directly attacks the fork class. Risk: low (CI-only).

### Phase 2 — `trait Executor` boundary + single `latest` impl + encapsulation test
Create the crates; **move the executor bodies** from `authority/src/execution/`
into `soma-execution-latest`; `authority` calls through `soma_execution`. The 5
inline `>= 1` gates move with the code and still select behavior within `latest`
(boundary is structural now; version-cutting comes at Phase 5). Risk: low-medium
(pure move). Verify: full test suite + protocol-config snapshots unchanged +
effects-digest determinism (before/after identical) + encapsulation test green.
Note: a dependency cycle forbids a pass-through delegating back to `authority` —
the code must actually move.

### Phase 3 — carve effect logic out of `types/` (no effect change, large)
Module-by-module pure-move PRs, least-coupled first, each gated by the
determinism check:
1. `accumulator.rs` `derive_id` (small but consensus-critical — validate the pattern)
2. `emission.rs::advance_epoch`
3. `staking.rs` F1 / `auto_settle`
4. `validator.rs::advance_epoch` + helpers (split: structs stay, methods move)
5. `system_state/mod.rs::advance_epoch` family
6. `channel.rs`/`provider.rs`/`offering.rs` mutating methods
7. `temporary_store.rs::into_effects`/`get_object_changes`/folding (last — most entangled;
   likely an extension trait impl'd in `latest`).
Tighten the encapsulation test to forbid effect-mutating methods in `types/`.
Risk: medium-high — this is where a wrong move forks. Mitigated by pre-mainnet
status + the determinism test + regenesis as backstop.

### Phase 4 — (optional) prove version-keyed dispatch under msim before genesis
A `#[cfg(msim)]` throwaway cut to exercise the dispatcher selecting by
`execution_version`. De-risks Phase 5.

### Phase 5 — take the `v0` cut at mainnet genesis (the first real freeze)
Copy `latest` → frozen `soma-execution-v0`; dispatcher `0 => v0, _ => latest`;
pin genesis `execution_version = 0`. From here the rule flips: `v0` immutable;
new effect changes go into `latest` behind a new `execution_version` + (for big
changes) a new cut. Risk: medium — a wrong dispatcher mapping forks. Verify:
golden-effects replay of genesis + dispatcher unit test.

## Pre-mainnet vs can-wait
- Must precede mainnet: Phase 1 (now), Phase 2, **Phase 3 (must land before the v0
  cut, or v0 freezes a still-mutating `types` and the pattern is defeated)**, Phase 5
  (the cut *is* genesis).
- Optional: Phase 4.
- Because `latest` is freely mutable pre-genesis, Phases 2–4 need not be
  strictly effect-preserving — but treat them as such (determinism test) to catch
  accidental divergence early.

## Riskiest spots (where a wrong move forks)
1. The `types/` carve-out boundary (Phase 3) — move only mutating methods; keep
   serde+getters; verify every consumer builds.
2. `accumulator.rs::derive_id` — a one-byte divergence forks all settlement.
   Move first, in isolation, carrying its determinism tests + a golden-ID assertion.
3. `into_effects`/`get_object_changes` ordering — any iteration/fold/lamport change
   changes the effects digest. Do last, behind a broad-corpus determinism test.
4. The dispatcher mapping at the cut (Phase 5) — unit test + golden genesis replay.
5. The `advance_epoch` family — most effect-sensitive code on the chain; move
   validator/staking/emission as isolated digest-verified PRs before touching it.

Safety property: if the v0 cut is last and is a byte-copy of a `latest` kept
effect-stable by the determinism test throughout, **no phase can fork mainnet** —
mainnet doesn't exist until the cut, and after it v0 is frozen.
