# CLI, SDK & RPC Crates — Comprehensive Testing Plan

Testing plan for `cli/src/`, `sdk/src/`, and `rpc/src/` achieving high parity with Sui's CLI (`MystenLabs/sui` — `crates/sui/src/`) and Sui's RPC SDK (`MystenLabs/sui-rust-sdk` — `crates/sui-rpc/`, `crates/sui-sdk-types/`, `crates/sui-crypto/`, `crates/sui-transaction-builder/`). Covers file-by-file mapping, attribution requirements, test infrastructure, and every test needed for parity plus Soma-specific coverage.

**Sui CLI reference**: `MystenLabs/sui` — `crates/sui/src/`
**Sui RPC SDK reference**: `MystenLabs/sui-rust-sdk` — `crates/sui-rpc/`, `crates/sui-sdk-types/`
**Soma crates**: `cli/src/`, `sdk/src/`, `rpc/src/`

---

## Audit Notes (Feb 2026)

**Priority Ranking**: #6 of 7 plans — user-facing but not consensus-critical. RPC/CLI bugs cause poor UX, not chain safety issues.

**Accuracy Correction**: The executive summary claims **32 existing tests**, but codebase verification found **29 actual test functions** (2 CLI + 18 SDK proxy_client + 7 RPC field masks + 2 RPC serde = 29). The 3-test discrepancy likely comes from counting proto validation tests (`file_descriptor_set_is_valid`) that are compile-time assertions rather than functional tests.

**Corrected total: ~29 existing tests, not 32.**

**Key Concerns**:
1. **RPC proto roundtrip tests (Priority 9) are the highest-value subset** — `rpc_proto_conversions.rs` is ~27k tokens with zero tests. A single conversion bug causes every RPC client to see corrupted data. Proptest-based roundtrip testing is the right approach.
2. **Stale Sui references in doc comments** — 8 files still reference "Sui blockchain", "sui_sdk_types", "Sui address" etc. These should be cleaned up as part of attribution work, not just noted.
3. **This plan is the most ambitious** — 163 new tests across 3 crates is a lot. Consider phasing: RPC proto roundtrips first, then SDK config, then CLI integration tests last.
4. **CLI integration tests (Priority 4) require msim** — 18 simtest tests are expensive to write and run. These should be deprioritized relative to RPC type safety tests.
5. **SDK proxy_client already has good coverage** — 18 tests is solid for a Soma-original module. The 5 new tests proposed are reasonable extensions.

**Estimated Effort**: ~17 engineering days as planned. Recommend splitting into two phases: RPC type safety (days 1-7) and CLI/SDK (days 8-17).

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [CLI Crate — File-by-File Cross-Reference](#cli-crate--file-by-file-cross-reference)
3. [SDK Crate — File-by-File Cross-Reference](#sdk-crate--file-by-file-cross-reference)
4. [RPC Crate — File-by-File Cross-Reference](#rpc-crate--file-by-file-cross-reference)
5. [Attribution Requirements](#attribution-requirements)
6. [Test Infrastructure](#test-infrastructure)
7. [Priority 1: CLI Pure Function Unit Tests](#priority-1-cli-pure-function-unit-tests)
8. [Priority 2: CLI Keytool Tests](#priority-2-cli-keytool-tests)
9. [Priority 3: CLI Response/Display Tests](#priority-3-cli-responsedisplay-tests)
10. [Priority 4: CLI Command Integration Tests (simtest)](#priority-4-cli-command-integration-tests-simtest)
11. [Priority 5: CLI Genesis Ceremony Tests](#priority-5-cli-genesis-ceremony-tests)
12. [Priority 6: SDK Client Config Tests](#priority-6-sdk-client-config-tests)
13. [Priority 7: SDK Error & Builder Tests](#priority-7-sdk-error--builder-tests)
14. [Priority 8: SDK Wallet Context Tests](#priority-8-sdk-wallet-context-tests)
15. [Priority 9: RPC Proto Roundtrip Tests (proptest)](#priority-9-rpc-proto-roundtrip-tests-proptest)
16. [Priority 10: RPC Type Serialization Tests](#priority-10-rpc-type-serialization-tests)
17. [Priority 11: RPC Proto Conversion Tests](#priority-11-rpc-proto-conversion-tests)
18. [Priority 12: RPC Field Mask Tests (expand existing)](#priority-12-rpc-field-mask-tests-expand-existing)
19. [Priority 13: RPC gRPC Handler Tests](#priority-13-rpc-grpc-handler-tests)
20. [Priority 14: RPC Client Tests](#priority-14-rpc-client-tests)
21. [Priority 15: RPC Serde & Utilities Tests](#priority-15-rpc-serde--utilities-tests)
22. [Implementation Order](#implementation-order)
23. [Build & Run Commands](#build--run-commands)

---

## Executive Summary

### Current State

| Crate | Files | Lines | Existing Tests | Test Coverage |
|-------|-------|-------|----------------|---------------|
| CLI | 26 | ~6,300 | 2 | Extremely low — only `format_soma()` and `truncate_address()` |
| SDK | 6 | ~1,640 | 18 | All in `proxy_client.rs`; 5 of 6 files have 0 tests |
| RPC | ~75 | ~15,000+ | 12 | Field masks + serde + proto validation only |
| **Total** | **~107** | **~23,000+** | **32** | **Critical gap** |

### Target State

| Crate | Target Tests | Gap |
|-------|-------------|-----|
| CLI | ~70 | ~68 new |
| SDK | ~45 | ~27 new |
| RPC | ~80 | ~68 new |
| **Total** | **~195** | **~163 new** |

### Test Count Summary

| Category | Sui Tests (est.) | Soma Existing | Gap |
|----------|-----------------|---------------|-----|
| CLI pure function unit tests | N/A | 2 | ~20 new |
| CLI keytool tests | 12 | 0 | ~10 adapted |
| CLI response/display tests | Snapshot-based | 0 | ~10 new |
| CLI command integration (simtest) | 65 | 0 | ~18 adapted |
| CLI genesis ceremony | 1 | 0 | 1 adapted |
| SDK client config | Sui has 0 inline | 0 | ~8 new |
| SDK error & builder | Sui has 0 inline | 0 | ~6 new |
| SDK wallet context | Sui has 0 inline | 0 | ~8 new |
| SDK proxy client | N/A (Soma-only) | 18 | 5 new |
| RPC proto roundtrip (proptest) | 12 | 0 | ~15 new |
| RPC type serialization | 87 (proptest) | 0 | ~20 adapted |
| RPC proto conversions | Part of proptest | 0 | ~12 new |
| RPC field mask tests | 7 | 7 | ~5 expand |
| RPC gRPC handler tests | N/A | 0 | ~10 new |
| RPC client tests | Integration only | 0 | ~8 new |
| RPC serde & utilities | 2 | 5 | ~5 expand |
| **Total** | **~185** | **32** | **~163** |

---

## CLI Crate — File-by-File Cross-Reference

### Legend
- **Very High** = Near-direct port from Sui, needs full attribution
- **High** = Heavily derived with Soma modifications, needs attribution
- **Medium** = Sui-inspired patterns, needs attribution
- **Low** = Sui-inspired concept, Soma implementation
- **None** = Entirely Soma-original, no attribution needed

### Command Infrastructure

| Soma File | Sui File | Derivation | Inline Tests | Notes |
|-----------|----------|------------|--------------|-------|
| `cli/src/main.rs` | `crates/sui/src/main.rs` | Very High | 0 | Has `Copyright Mysten Labs` header. `exit_main!` macro, clap, tracing |
| `cli/src/lib.rs` | `crates/sui/src/lib.rs` | Medium | 0 | Module declarations follow Sui pattern |
| `cli/src/soma_commands.rs` | `crates/sui/src/sui_commands.rs` | High | 0 | `SomaCommand` enum, `start()`, `genesis()`, `prompt_if_no_config()`, `get_wallet_context()` |
| `cli/src/client_commands.rs` | `crates/sui/src/client_commands.rs` | High | 0 | `execute_or_serialize()`, `TxProcessingArgs`, Base64 patterns |
| `cli/src/keytool.rs` | `crates/sui/src/keytool.rs` | Very High | 0 | Near-direct port: `KeyToolCommand`, `Key`, `CommandOutput` |
| `cli/src/response.rs` | Sui display patterns | Medium | **2** | `format_soma()`, `truncate_address()` tested |

### Command Modules — Sui-Derived

| Soma File | Sui File | Derivation | Inline Tests | Notes |
|-----------|----------|------------|--------------|-------|
| `cli/src/commands/validator.rs` | `crates/sui/src/validator_commands.rs` | High | 0 | `make_validator_info`, join/leave committee, metadata update |
| `cli/src/genesis_ceremony/mod.rs` | `crates/sui/src/genesis_ceremony.rs` | High | 0 | Ceremony workflow; Soma adds `add-model` |
| `cli/src/genesis_ceremony/genesis_inspector.rs` | `crates/sui/src/genesis_inspector.rs` | High | 0 | Interactive inspector with inquire |
| `cli/src/commands/env.rs` | Sui env patterns | Medium-High | 0 | Multi-environment config management |
| `cli/src/commands/wallet.rs` | Sui address mgmt | Medium-High | 0 | Address management |
| `cli/src/commands/objects.rs` | Sui object queries | Medium | 0 | Object listing, gas coin queries |
| `cli/src/commands/send.rs` | Sui transfer concept | Low-Medium | 0 | TransferCoin via TransactionKind |
| `cli/src/commands/transfer.rs` | Sui transfer-object | Low-Medium | 0 | TransferObjects |
| `cli/src/commands/pay.rs` | Sui pay command | Low-Medium | 0 | PayCoins |
| `cli/src/commands/stake.rs` | Sui staking concept | Low-Medium | 0 | Extends to model staking |

### Command Modules — Soma-Original (No Attribution)

| Soma File | Inline Tests | Notes |
|-----------|--------------|-------|
| `cli/src/commands/mod.rs` | 0 | Module declarations only |
| `cli/src/commands/model.rs` | 0 | 708 lines — model commit/reveal/update/deactivate/info/list |
| `cli/src/commands/target.rs` | 0 | Target list/info |
| `cli/src/commands/challenge.rs` | 0 | Challenge initiate/info/list |
| `cli/src/commands/submit.rs` | 0 | Data submission |
| `cli/src/commands/claim.rs` | 0 | Reward claiming |
| `cli/src/commands/data.rs` | 0 | Data download via proxy |
| `cli/src/commands/balance.rs` | 0 | Balance queries |
| `cli/src/commands/tx.rs` | 0 | Transaction queries |

### Sui CLI Files With No Soma Counterpart

| Sui File | Reason Not Needed |
|----------|-------------------|
| `src/client_ptb/` (lexer, parser, builder, errors) | Soma has no PTB / Move VM |
| `src/clever_error_rendering.rs` | Move abort error rendering |
| `src/upgrade_compatibility/` | Move package upgrades |
| `src/mvr_resolver.rs` | Move Version Registry |
| `src/fire_drill.rs` | Sui-specific fire drill |
| `src/zklogin_commands_util.rs` | zkLogin (not in Soma) |
| `src/external_signer.rs` | External signer (not in Soma) |
| `src/trace_analysis_commands.rs` | Trace analysis |
| `tests/shell_tests/` (~70 .sh files) | Shell snapshot tests (Soma could adopt pattern later) |
| `tests/ptb_files/` (~52 .ptb files) | PTB file tests (N/A) |

---

## SDK Crate — File-by-File Cross-Reference

| Soma File | Sui Equivalent | Derivation | Existing Tests | Notes |
|-----------|---------------|------------|----------------|-------|
| `sdk/src/lib.rs` | Sui SDK client pattern | Partial | 0 | `SomaClient`/`SomaClientBuilder` wrapping `Arc<RwLock<Client>>` |
| `sdk/src/client_config.rs` | `SuiClientConfig`/`SuiEnv` | Heavy | 0 | Direct adaptation with gRPC instead of JSON-RPC |
| `sdk/src/error.rs` | Sui SDK error types | Heavy | 0 | Error variants mirror Sui almost 1:1 |
| `sdk/src/transaction_builder.rs` | Sui TransactionBuilder | Partial | 0 | Much simpler (no Move VM), gas-selection pattern similar |
| `sdk/src/wallet_context.rs` | Sui WalletContext | Heavy | 0 | Near-direct port with gRPC streaming |
| `sdk/src/proxy_client.rs` | N/A (Soma-only) | None | **18** | Validator proxy for data/model fetching |

---

## RPC Crate — File-by-File Cross-Reference

### API Layer

| Soma File | sui-rust-sdk File | Derivation | Existing Tests | Notes |
|-----------|-------------------|------------|----------------|-------|
| `rpc/src/api/mod.rs` | `sui-rpc/src/lib.rs` | Heavy | 0 | Router + service registration |
| `rpc/src/api/client.rs` | `sui-rpc/src/client/mod.rs` | Heavy | 0 | High-level client with 20+ methods |
| `rpc/src/api/reader.rs` | Sui StateReader pattern | Heavy | 0 | `StateReader` wrapping trait |
| `rpc/src/api/response.rs` | Sui response headers | Medium | 0 | X-Soma-* headers (was X-Sui-*) |
| `rpc/src/api/error.rs` | Sui RPC errors | Heavy | 0 | Error conversions |
| `rpc/src/api/subscription.rs` | Sui subscription | Heavy | 0 | Checkpoint streaming |

### RPC Client Submodule

| Soma File | sui-rust-sdk File | Derivation | Existing Tests | Notes |
|-----------|-------------------|------------|----------------|-------|
| `rpc/src/api/rpc_client/mod.rs` | `sui-rpc/src/client/mod.rs` | Heavy | 0 | Low-level tonic client |
| `rpc/src/api/rpc_client/headers.rs` | `sui-rpc/src/headers.rs` | Heavy | 0 | X-Soma-* constants |
| `rpc/src/api/rpc_client/lists.rs` | Sui pagination stream | Heavy | 0 | `list_owned_objects` pagination |
| `rpc/src/api/rpc_client/interceptors.rs` | Sui interceptors | Heavy | 0 | Basic/bearer auth |
| `rpc/src/api/rpc_client/response_ext.rs` | Sui response_ext | Heavy | 0 | Still says "Sui specific data" in docstring |
| `rpc/src/api/rpc_client/transaction_execution.rs` | Sui tx execution client | Heavy | 0 | Checkpoint watching |

### gRPC Service Handlers — Ledger Service

| Soma File | sui-rust-sdk Equivalent | Derivation | Existing Tests | Notes |
|-----------|------------------------|------------|----------------|-------|
| `rpc/src/api/grpc/ledger_service/mod.rs` | Sui LedgerService | Heavy | 0 | Dispatch to handlers |
| `rpc/src/api/grpc/ledger_service/get_object.rs` | Sui get_object handler | Heavy | 0 | **Copyright Mysten Labs** |
| `rpc/src/api/grpc/ledger_service/get_transaction.rs` | Sui get_transaction | Heavy | 0 | **Copyright Mysten Labs** |
| `rpc/src/api/grpc/ledger_service/get_checkpoint.rs` | Sui get_checkpoint | Heavy | 0 | |
| `rpc/src/api/grpc/ledger_service/get_epoch.rs` | Sui get_epoch | Heavy | 0 | **Copyright Mysten Labs** |
| `rpc/src/api/grpc/ledger_service/get_service_info.rs` | Sui get_service_info | Heavy | 0 | |

### gRPC Service Handlers — State Service

| Soma File | sui-rust-sdk Equivalent | Derivation | Existing Tests | Notes |
|-----------|------------------------|------------|----------------|-------|
| `rpc/src/api/grpc/state_service/mod.rs` | — | Mixed | 0 | |
| `rpc/src/api/grpc/state_service/get_balance.rs` | Sui get_balance | Heavy | 0 | **Copyright Mysten Labs** |
| `rpc/src/api/grpc/state_service/list_owned_objects.rs` | Sui list_owned_objects | Heavy | 0 | |
| `rpc/src/api/grpc/state_service/get_target.rs` | N/A | Soma-only | 0 | **Copyright Soma Foundation** |
| `rpc/src/api/grpc/state_service/get_challenge.rs` | N/A | Soma-only | 0 | **Copyright Soma Foundation** |
| `rpc/src/api/grpc/state_service/list_targets.rs` | N/A | Soma-only | 0 | **Copyright Soma Foundation** |
| `rpc/src/api/grpc/state_service/list_challenges.rs` | N/A | Soma-only | 0 | **Copyright Soma Foundation** |

### gRPC Service Handlers — Transaction Execution & Subscription

| Soma File | Derivation | Existing Tests | Notes |
|-----------|------------|----------------|-------|
| `rpc/src/api/grpc/transaction_execution_service/mod.rs` | Heavy | 0 | |
| `rpc/src/api/grpc/transaction_execution_service/simulate.rs` | Heavy | 0 | |
| `rpc/src/api/grpc/subscription_service.rs` | Heavy | 0 | |

### Types Module (SDK-level types, derived from sui-sdk-types)

| Soma File | sui-sdk-types File | Derivation | Existing Tests | Notes |
|-----------|-------------------|------------|----------------|-------|
| `rpc/src/types/transaction.rs` | `src/transaction/` | Heavy + Soma extensions | 0 | All 31 Soma tx variants |
| `rpc/src/types/object.rs` | `src/object.rs` | Heavy + Soma extensions | 0 | Target/Submission/Challenge types |
| `rpc/src/types/effects.rs` | `src/effects/` | Heavy + Soma extensions | 0 | `TransactionFee` instead of gas |
| `rpc/src/types/execution_status.rs` | `src/execution_status.rs` | Heavy + Soma extensions | 0 | Soma-specific error variants |
| `rpc/src/types/fee.rs` | N/A | Soma-only | 0 | 4-part fee struct |
| `rpc/src/types/checkpoint.rs` | `src/checkpoint.rs` | Heavy | 0 | |
| `rpc/src/types/digest.rs` | `src/digest.rs` | Heavy | 0 | Base58 |
| `rpc/src/types/address.rs` | `src/address.rs` | Heavy | 0 | Hex with 0x prefix |
| `rpc/src/types/hash.rs` | `src/hash.rs` | Heavy | 0 | Doc refs still say `sui_sdk_types` |
| `rpc/src/types/balance_change.rs` | Similar | Heavy | 0 | |
| `rpc/src/types/bitmap.rs` | `src/bitmap.rs` | Heavy | 0 | Doc refs still say `sui_sdk_types` |
| `rpc/src/types/crypto/mod.rs` | `src/crypto/` | Heavy | 0 | |
| `rpc/src/types/crypto/signature.rs` | `src/crypto/signature.rs` | Heavy | 0 | |
| `rpc/src/types/crypto/multisig.rs` | `src/crypto/multisig.rs` | Heavy | 0 | |

### Utils Module

| Soma File | sui-rust-sdk File | Derivation | Existing Tests | Notes |
|-----------|-------------------|------------|----------------|-------|
| `rpc/src/utils/merge.rs` | `sui-rpc/src/merge.rs` | Heavy | 0 | `Merge` trait |
| `rpc/src/utils/rpc_proto_conversions.rs` | `sui-rpc/src/proto/` conversions | Heavy + Soma | 0 | **Largest file** (~27k tokens), 0 tests |
| `rpc/src/utils/types_conversions.rs` | Sui type conversions | Heavy | 0 | |
| `rpc/src/utils/field/mod.rs` | `sui-rpc/src/field/mod.rs` | Heavy | **2** | Path validation |
| `rpc/src/utils/field/field_mask_tree.rs` | `sui-rpc/src/field/field_mask_tree.rs` | Heavy | **2** | Tree operations |
| `rpc/src/utils/field/field_mask_util.rs` | `sui-rpc/src/field/field_mask_util.rs` | Heavy | **3** | Normalize/validate |
| `rpc/src/utils/_serde/mod.rs` | `sui-rpc/src/_serde/mod.rs` | Heavy | **2** | Base64 + value deser |
| `rpc/src/utils/_serde/well_known_types.rs` | `sui-rpc/src/_serde/well_known_types.rs` | Heavy | 0 | Protobuf WKT serde |

### Proto Module

| Soma File | Derivation | Existing Tests | Notes |
|-----------|------------|----------------|-------|
| `rpc/src/proto/mod.rs` | Heavy | 0 | `TryFromProtoError`, timestamp helpers |
| `rpc/src/proto/soma/mod.rs` | Heavy | **1** | `file_descriptor_set_is_valid` |
| `rpc/src/proto/google.rs` | Heavy | **2** | FDS validation x2 |
| `rpc/src/proto/generated/soma.rpc.rs` | Generated | 0 | Still references "Sui blockchain" and "Sui address" in comments |

---

## Attribution Requirements

### Attribution Header Template

For files derived from `MystenLabs/sui` (CLI, SDK):
```
// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui/src/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
```

For files derived from `MystenLabs/sui-rust-sdk` (RPC):
```
// Portions of this file are derived from sui-rust-sdk (MystenLabs/sui-rust-sdk).
// Original source: https://github.com/MystenLabs/sui-rust-sdk/tree/master/crates/sui-rpc/src/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
```

### CLI Crate — Files Requiring Attribution (14 files)

**From `MystenLabs/sui` — `crates/sui/src/`:**

| Soma File | Sui Reference | Derivation |
|-----------|--------------|------------|
| `cli/src/main.rs` | `crates/sui/src/main.rs` | Very High (already has Mysten copyright) |
| `cli/src/keytool.rs` | `crates/sui/src/keytool.rs` | Very High |
| `cli/src/soma_commands.rs` | `crates/sui/src/sui_commands.rs` | High |
| `cli/src/client_commands.rs` | `crates/sui/src/client_commands.rs` | High |
| `cli/src/commands/validator.rs` | `crates/sui/src/validator_commands.rs` | High |
| `cli/src/genesis_ceremony/mod.rs` | `crates/sui/src/genesis_ceremony.rs` | High |
| `cli/src/genesis_ceremony/genesis_inspector.rs` | `crates/sui/src/genesis_inspector.rs` | High |
| `cli/src/commands/env.rs` | Sui env patterns | Medium-High |
| `cli/src/commands/wallet.rs` | Sui address mgmt | Medium-High |
| `cli/src/lib.rs` | `crates/sui/src/lib.rs` | Medium |
| `cli/src/response.rs` | Sui display patterns | Medium |
| `cli/src/commands/objects.rs` | Sui object queries | Medium |
| `cli/src/commands/send.rs` | Sui transfer concept | Low-Medium |
| `cli/src/commands/transfer.rs` | Sui transfer-object | Low-Medium |

### CLI Crate — Files NOT Requiring Attribution (11 files)

`commands/mod.rs`, `commands/model.rs`, `commands/target.rs`, `commands/challenge.rs`, `commands/submit.rs`, `commands/claim.rs`, `commands/data.rs`, `commands/balance.rs`, `commands/tx.rs`, `commands/pay.rs`, `commands/stake.rs`

### SDK Crate — Files Requiring Attribution (4 files)

**From `MystenLabs/sui` — `crates/sui-sdk/`:**

| Soma File | Sui Reference | Derivation |
|-----------|--------------|------------|
| `sdk/src/client_config.rs` | `SuiClientConfig` / `SuiEnv` | Heavy |
| `sdk/src/error.rs` | Sui SDK error types | Heavy |
| `sdk/src/wallet_context.rs` | Sui WalletContext | Heavy |
| `sdk/src/lib.rs` | Sui SDK client pattern | Partial |

### SDK Crate — Files NOT Requiring Attribution (2 files)

`sdk/src/proxy_client.rs` (Soma-original), `sdk/src/transaction_builder.rs` (partially derived but significantly simplified)

### RPC Crate — Files Requiring Attribution (~40+ files)

**Files with existing `Copyright Mysten Labs` headers (already attributed):**
- `rpc/src/api/grpc/ledger_service/get_epoch.rs`
- `rpc/src/api/grpc/ledger_service/get_object.rs`
- `rpc/src/api/grpc/ledger_service/get_transaction.rs`
- `rpc/src/api/grpc/state_service/get_balance.rs`

**Files needing attribution added (from `MystenLabs/sui-rust-sdk`):**

*API Layer (12 files):*
`api/mod.rs`, `api/client.rs`, `api/reader.rs`, `api/response.rs`, `api/error.rs`, `api/subscription.rs`, `api/rpc_client/mod.rs`, `api/rpc_client/headers.rs`, `api/rpc_client/lists.rs`, `api/rpc_client/interceptors.rs`, `api/rpc_client/response_ext.rs`, `api/rpc_client/transaction_execution.rs`

*gRPC Handlers (8 files):*
`api/grpc/ledger_service/mod.rs`, `get_checkpoint.rs`, `get_service_info.rs`, `api/grpc/state_service/mod.rs`, `list_owned_objects.rs`, `api/grpc/transaction_execution_service/mod.rs`, `simulate.rs`, `api/grpc/subscription_service.rs`

*Types (14 files):*
All files in `rpc/src/types/` (derived from `sui-sdk-types`)

*Utils (7 files):*
`utils/merge.rs`, `utils/rpc_proto_conversions.rs`, `utils/types_conversions.rs`, `utils/field/mod.rs`, `utils/field/field_mask_tree.rs`, `utils/field/field_mask_util.rs`, `utils/_serde/mod.rs`, `utils/_serde/well_known_types.rs`

*Proto (2 files):*
`proto/mod.rs`, `proto/soma/mod.rs`

### RPC Crate — Files NOT Requiring Attribution (Soma-original, 4 files)

`api/grpc/state_service/get_target.rs`, `api/grpc/state_service/get_challenge.rs`, `api/grpc/state_service/list_targets.rs`, `api/grpc/state_service/list_challenges.rs`

### Additional Cleanup: Stale Sui References

These files still reference Sui in doc comments / generated code and should be updated:

| File | Issue |
|------|-------|
| `rpc/src/types/hash.rs` | Doc examples reference `sui_sdk_types::Address`, `sui_sdk_types::Ed25519PublicKey` |
| `rpc/src/types/bitmap.rs` | Doc examples reference `sui_sdk_types::Bitmap` (11 occurrences) |
| `rpc/src/api/rpc_client/response_ext.rs` | Docstring: "Sui specific data" |
| `rpc/src/utils/field/field_mask_tree.rs` | Example: `# use sui_rpc::field::FieldMaskTree;` |
| `rpc/src/types/crypto/signature.rs` | "schemes supported by Sui" |
| `rpc/src/types/crypto/multisig.rs` | "Sui blockchain" |
| `rpc/src/types/checkpoint.rs` | "state of the Sui blockchain" |
| `rpc/src/proto/generated/soma.rpc.rs` | "An object on the Sui blockchain", "owner's Sui address", "Sui" in subscription descriptions |

---

## Test Infrastructure

### Existing Infrastructure

**Test Cluster** (`test-cluster/src/lib.rs`):
```rust
let cluster = TestClusterBuilder::new()
    .with_num_validators(4)
    .with_epoch_duration_ms(5000)
    .build()
    .await;
```

**Wallet Context for Tests** (`sdk/src/wallet_context.rs`):
```rust
WalletContext::new_for_tests(config_path, request_timeout, max_concurrent_requests)
```

### Infrastructure to Create

#### 1. CLI Test Utilities Module

**File**: `cli/src/unit_tests/mod.rs`
```rust
#[cfg(test)]
mod cli_unit_tests;
```

**File**: `cli/src/unit_tests/cli_unit_tests.rs`

Shared test helpers:
- `create_test_wallet_context()` — Create a WalletContext with temp keystore
- `create_test_config()` — Create a SomaClientConfig with temp directory
- `assert_command_output_contains(output, expected)` — Output assertion helper

#### 2. RPC Proptest Infrastructure

**File**: `rpc/src/utils/proptests.rs`

Following Sui's `protobuf_roundtrip_test!` macro pattern:
```rust
macro_rules! protobuf_roundtrip_test {
    ($test_name:ident, $domain_type:ty, $proto_type:ty) => {
        #[test_strategy::proptest]
        fn $test_name(value: $domain_type) {
            // Domain -> Proto -> encode -> decode -> Proto -> Domain
            let proto: $proto_type = (&value).into();
            let bytes = proto.encode_to_vec();
            let decoded = <$proto_type>::decode(&*bytes).unwrap();
            let roundtripped: $domain_type = (&decoded).try_into().unwrap();
            assert_eq!(value, roundtripped);
        }
    };
}
```

Requires adding `proptest` and `test-strategy` as dev-dependencies to the `rpc` crate, plus implementing `Arbitrary` for Soma domain types (or using the existing `proptest` feature if available).

#### 3. RPC Mock StateReader for Handler Tests

**File**: `rpc/src/api/grpc/test_utils.rs`

Mock implementation of `RpcStateReader` trait for unit-testing gRPC handlers without a live node:
```rust
struct MockStateReader {
    objects: HashMap<ObjectID, Object>,
    system_state: Option<SystemState>,
    // ...
}
impl RpcStateReader for MockStateReader { ... }
```

---

## Priority 1: CLI Pure Function Unit Tests

**File to create**: `cli/src/unit_tests/pure_function_tests.rs`
**Sui equivalent**: No direct equivalent — these are Soma-specific helper functions

### Tests to Implement (~20 tests)

#### From `commands/model.rs`:

| Test | Description |
|------|-------------|
| `test_parse_hex_digest_32_valid` | Valid 64-char hex string with and without `0x` prefix |
| `test_parse_hex_digest_32_invalid_length` | Rejects non-32-byte hex |
| `test_parse_hex_digest_32_invalid_hex` | Rejects non-hex characters |
| `test_parse_hex_digest_32_empty` | Rejects empty string |
| `test_parse_embedding_valid` | CSV of floats: `"0.1,0.2,0.3"` → `SomaTensor` |
| `test_parse_embedding_empty` | Rejects empty embedding |
| `test_parse_embedding_invalid_float` | Rejects non-numeric values |
| `test_build_weights_manifest_valid` | URL + checksum + size + key → `ModelWeightsManifest` |
| `test_truncate_id_short` | Strings shorter than truncation point unchanged |
| `test_truncate_id_long` | Long strings truncated with `...` |
| `test_model_to_summary_conversion` | Model → ModelSummary field mapping |
| `test_commission_rate_validation` | Rate > 10000 rejected |

#### From `commands/target.rs`:

| Test | Description |
|------|-------------|
| `test_format_target_status_open` | `TargetStatus::Open` → "Open" |
| `test_format_target_status_filled` | `TargetStatus::Filled { fill_epoch }` → "Filled (epoch N)" |
| `test_format_target_status_claimed` | `TargetStatus::Claimed` → "Claimed" |

#### From `commands/challenge.rs`:

| Test | Description |
|------|-------------|
| `test_format_challenge_status_pending` | `ChallengeStatus::Pending` → "Pending" |
| `test_format_challenge_status_resolved` | `ChallengeStatus::Resolved { challenger_lost }` formatting |

#### From `soma_commands.rs`:

| Test | Description |
|------|-------------|
| `test_socket_addr_to_url` | `SocketAddr` → URL string conversion |
| `test_normalize_bind_addr` | Bind address normalization |

#### From `commands/submit.rs`:

| Test | Description |
|------|-------------|
| `test_build_data_manifest` | URL + checksum → `SubmissionManifest` |

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `cli/src/commands/model.rs` | N/A (Soma-original) |
| `cli/src/commands/target.rs` | N/A (Soma-original) |
| `cli/src/commands/challenge.rs` | N/A (Soma-original) |
| `cli/src/soma_commands.rs` | `crates/sui/src/sui_commands.rs` |

---

## Priority 2: CLI Keytool Tests

**File to create**: `cli/src/unit_tests/keytool_tests.rs`
**Sui equivalent**: `crates/sui/src/unit_tests/keytool_tests.rs` (12 tests)

### Sui's Keytool Tests (for parity reference)

| # | Sui Test | Soma Relevance | Description |
|---|----------|----------------|-------------|
| 1 | `test_addresses_command` | High | List addresses with Ed25519 and Secp256k1 keypairs |
| 2 | `test_flag_in_signature_and_keypair` | High | Signature scheme flag correctness for all key types |
| 3 | `test_read_write_keystore_with_flag` | High | Reading/writing keys to file with flag byte |
| 4 | `test_sui_operations_config` | Medium | Hardcoded keystore address derivation |
| 5 | `test_load_keystore_err` | High | Error on loading authority keypair without flag byte |
| 6 | `test_private_keys_import_export` | High | Bech32/Hex/Base64 private key import, export, malformed rejection |
| 7 | `test_mnemonics_ed25519` | High | Mnemonic-to-Ed25519 key derivation |
| 8 | `test_mnemonics_secp256k1` | High | Mnemonic-to-Secp256k1 key derivation |
| 9 | `test_mnemonics_secp256r1` | Medium | Mnemonic-to-Secp256r1 key derivation |
| 10 | `test_invalid_derivation_path` | High | Rejection of invalid BIP44/BIP54 derivation paths |
| 11 | `test_valid_derivation_path` | High | Acceptance of valid derivation paths |
| 12 | `test_keytool_bls12381` | Medium | BLS12381 key generation |
| 13 | `test_sign_command` | High | Signing with address or alias, custom intents |

### Tests to Implement for Soma (~10 tests)

| Test | Description | Adapted From |
|------|-------------|-------------|
| `test_keytool_generate_ed25519` | Generate Ed25519 key, verify address derivation | Sui #7 |
| `test_keytool_generate_secp256k1` | Generate Secp256k1 key, verify address derivation | Sui #8 |
| `test_keytool_import_export_round_trip` | Import private key (somaprivkey Bech32), export, compare | Sui #6 |
| `test_keytool_import_hex_key` | Import from hex, verify address matches | Sui #6 |
| `test_keytool_flag_correctness` | Verify signature flag bytes for each scheme | Sui #2 |
| `test_keytool_list_addresses` | Generate multiple keys, list all addresses | Sui #1 |
| `test_keytool_sign_verify` | Sign a message, verify signature | Sui #13 |
| `test_keytool_invalid_derivation_path` | Reject invalid BIP44 paths | Sui #10 |
| `test_keytool_multisig_address` | MultiSig address derivation and combine | Soma-specific |
| `test_keytool_load_keystore_error` | Error on corrupt keystore | Sui #5 |

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `cli/src/keytool.rs` | `crates/sui/src/keytool.rs` |
| `cli/src/unit_tests/keytool_tests.rs` (create) | `crates/sui/src/unit_tests/keytool_tests.rs` |

---

## Priority 3: CLI Response/Display Tests

**File to create**: `cli/src/unit_tests/response_tests.rs`
**Sui equivalent**: Sui uses insta snapshots; Soma should add assertion-based tests

### Tests to Implement (~10 tests, expanding existing 2)

| Test | Description |
|------|-------------|
| `test_format_soma_zero` | `format_soma(0)` → "0 SOMA" |
| `test_format_soma_sub_soma` | `format_soma(999_999_999)` → "0.999999999 SOMA" |
| `test_format_soma_exact` | `format_soma(1_000_000_000)` → "1 SOMA" |
| `test_format_soma_large` | `format_soma(1_234_567_890_123)` → "1234.567890123 SOMA" |
| `test_truncate_address_full` | Full 66-char hex address truncated correctly |
| `test_truncate_address_short` | Short address unchanged |
| `test_truncate_key_display` | Base64 key truncation |
| `test_transaction_status_is_success` | `TransactionStatus::Success` → true |
| `test_transaction_status_is_failure` | `TransactionStatus::Failure` → false |
| `test_owner_display_from_address_owner` | `Owner::AddressOwner` → hex display |
| `test_owner_display_from_shared` | `Owner::Shared` → "Shared" |
| `test_owner_display_from_immutable` | `Owner::Immutable` → "Immutable" |

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `cli/src/response.rs` | Sui display patterns (snapshot-based) |

---

## Priority 4: CLI Command Integration Tests (simtest)

**File to create**: `e2e-tests/tests/cli_tests.rs`
**Sui equivalent**: `crates/sui/tests/cli_tests.rs` (65 sim_test tests)

These tests require `RUSTFLAGS="--cfg msim"` and use the `TestCluster`.

### Sui Tests Adapted for Soma (~18 tests)

| Test | Sui Equivalent | Description |
|------|---------------|-------------|
| `test_genesis` | `test_genesis` | Genesis creates valid network config |
| `test_balance_command` | `test_gas_command` | Balance query returns correct amount |
| `test_objects_list_command` | `test_objects_command` | List owned objects |
| `test_object_info_command` | `test_object_info_get_command` | Get specific object by ID |
| `test_send_command` | `test_native_transfer` | Transfer SOMA tokens |
| `test_transfer_command` | `test_transfer` | Transfer arbitrary objects |
| `test_pay_command` | `test_pay` | Multi-recipient payment |
| `test_stake_with_validator` | `test_stake_with_u64_amount` | Stake SOMA with validator |
| `test_unstake_command` | N/A | Withdraw staked SOMA |
| `test_switch_env_command` | `test_switch_command` | Switch active environment |
| `test_new_address_command` | `test_new_address_command_by_flag` | Create new address |
| `test_execute_signed_tx` | `test_execute_signed_tx` | Execute pre-signed transaction |
| `test_serialize_tx` | `test_serialize_tx` | Transaction serialization round-trip |
| `test_model_commit_reveal` | N/A (Soma-only) | Model commit in epoch N, reveal in epoch N+1 |
| `test_target_list_command` | N/A (Soma-only) | List targets with status filter |
| `test_submit_and_claim` | N/A (Soma-only) | Submit data to target, wait, claim rewards |
| `test_challenge_initiate` | N/A (Soma-only) | Initiate challenge against filled target |
| `test_validator_report_model` | N/A (Soma-only) | Validator reports model unavailability |

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `e2e-tests/tests/cli_tests.rs` (create) | `crates/sui/tests/cli_tests.rs` |

---

## Priority 5: CLI Genesis Ceremony Tests

**File to create**: Inline test in `cli/src/genesis_ceremony/mod.rs`
**Sui equivalent**: `crates/sui/src/genesis_ceremony.rs` inline `ceremony` test

### Tests to Implement (1 test)

| Test | Description |
|------|-------------|
| `test_genesis_ceremony` | Full ceremony lifecycle: init → add validators → add models → build unsigned checkpoint → verify-and-sign for all validators → finalize. Verify genesis checkpoint is valid and contains expected models/validators. |

### Key Implementation Details from Sui

Sui's `ceremony` test:
- Creates temp directory
- Initializes ceremony with genesis config
- Adds 10 validators in a loop
- Builds unsigned checkpoint
- Each validator verifies and signs
- Finalizes → verifies output files exist

Soma adaptation:
- Same flow but also calls `add-model` for seed models
- Verify `SystemState` in genesis contains the models
- Verify `TargetState` has initial targets

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `cli/src/genesis_ceremony/mod.rs` | `crates/sui/src/genesis_ceremony.rs` |

---

## Priority 6: SDK Client Config Tests

**File to create**: Inline tests in `sdk/src/client_config.rs`
**Sui equivalent**: No inline tests in Sui's SDK, but the patterns are testable

### Tests to Implement (~8 tests)

| Test | Description |
|------|-------------|
| `test_config_new_empty` | `SomaClientConfig::new()` has no envs, no active address |
| `test_get_env_matching_alias` | `get_env("localnet")` returns matching env |
| `test_get_env_no_match` | `get_env("nonexistent")` returns None |
| `test_get_active_env_set` | Returns env when `active_env` is set |
| `test_get_active_env_none` | Error when `active_env` is None |
| `test_add_env_no_duplicate` | Adding same alias twice doesn't duplicate |
| `test_update_env_chain_id` | Chain ID update for existing env |
| `test_env_factory_methods` | `SomaEnv::localnet()`, `devnet()`, `testnet()` produce correct URLs |

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `sdk/src/client_config.rs` | Sui's `SuiClientConfig` in `sui-sdk` |

---

## Priority 7: SDK Error & Builder Tests

**File to create**: Inline tests in `sdk/src/error.rs` and `sdk/src/transaction_builder.rs`

### Error Tests (~3 tests)

| Test | Description |
|------|-------------|
| `test_error_display_rpc` | `Error::RpcError` display message |
| `test_error_display_insufficient_fund` | `Error::InsufficientFund` includes address and amounts |
| `test_error_display_server_version_mismatch` | Version mismatch includes client/server versions |

### TransactionBuilder Tests (~3 tests)

| Test | Description |
|------|-------------|
| `test_execution_options_defaults` | `ExecutionOptions::new()` defaults to no serialize, no gas |
| `test_execution_options_serialize` | `serialize_unsigned()` setter works |
| `test_execution_options_with_gas` | `with_gas()` setter stores object ref |

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `sdk/src/error.rs` | Sui SDK error types |
| `sdk/src/transaction_builder.rs` | Sui TransactionBuilder (much simpler) |

---

## Priority 8: SDK Wallet Context Tests

**File to create**: Inline tests in `sdk/src/wallet_context.rs`
**Sui equivalent**: Sui's WalletContext has no inline tests either, but is tested via CLI integration tests

### Tests to Implement (~8 tests)

| Test | Description |
|------|-------------|
| `test_wallet_context_new_for_tests` | `new_for_tests()` constructs valid context |
| `test_active_address_auto_set` | First address auto-set when none configured |
| `test_active_address_no_addresses` | Error when keystore is empty |
| `test_get_addresses_returns_all` | All keystore addresses returned |
| `test_has_addresses_true` | Returns true when addresses exist |
| `test_has_addresses_false` | Returns false when keystore empty |
| `test_get_one_address` | Returns first address |
| `test_env_override` | `with_env_override()` changes active env |

### Proxy Client Tests — Expand Existing (~5 new tests)

**File**: `sdk/src/proxy_client.rs` (18 existing tests)

| Test | Description |
|------|-------------|
| `test_fetch_url_path_submission` | Verify submission data URL path construction |
| `test_fetch_url_path_model` | Verify model weights URL path construction |
| `test_timeout_for_size_large` | Large data timeout calculation (e.g., 1GB) |
| `test_proxy_error_all_validators_failed` | Display for AllValidatorsFailed with details |
| `test_multiaddr_to_http_url_ip6` | IPv6 multiaddr conversion |

---

## Priority 9: RPC Proto Roundtrip Tests (proptest)

**File to create**: `rpc/src/utils/proptests.rs`
**Sui equivalent**: `crates/sui-rpc/src/proto/sui/rpc/v2/proptests.rs` (12 proptest roundtrip tests)

### Sui's Protobuf Roundtrip Tests

Sui tests these 12 types:
1. `CheckpointSummary`
2. `CheckpointContents`
3. `Transaction`
4. `TransactionEffects`
5. `ChangedObject`
6. `UnchangedConsensusObject`
7. `TransactionEvents`
8. `Object`
9. `UserSignature`
10. `ValidatorAggregatedSignature`
11. `ExecutionStatus`
12. `Transaction` (no BCS variant)

### Tests to Implement for Soma (~15 tests)

| Test | Description |
|------|-------------|
| `test_protobuf_roundtrip_transaction` | All 31 TransactionKind variants via proptest |
| `test_protobuf_roundtrip_transaction_effects` | TransactionEffects with Soma's TransactionFee |
| `test_protobuf_roundtrip_object` | Object with Soma types (Coin, Target, Submission, Challenge) |
| `test_protobuf_roundtrip_checkpoint_summary` | CheckpointSummary roundtrip |
| `test_protobuf_roundtrip_checkpoint_contents` | CheckpointContents roundtrip |
| `test_protobuf_roundtrip_execution_status` | All Soma-specific error variants |
| `test_protobuf_roundtrip_user_signature` | UserSignature roundtrip |
| `test_protobuf_roundtrip_validator_aggregated_signature` | ValidatorAggregatedSignature roundtrip |
| `test_protobuf_roundtrip_system_state` | SystemState with model registry, target state, emission pool |
| `test_protobuf_roundtrip_target` | Target with all fields (embedding, model_ids, etc.) |
| `test_protobuf_roundtrip_challenge` | Challenge with all fields |
| `test_protobuf_roundtrip_balance_change` | BalanceChange roundtrip |
| `test_protobuf_roundtrip_transaction_fee` | TransactionFee (base/operation/value fee) |
| `test_protobuf_roundtrip_epoch` | Epoch info roundtrip |
| `test_protobuf_roundtrip_transaction_no_bcs` | Transaction with BCS field stripped |

### Prerequisites

- Add `proptest` and `test-strategy` as dev-dependencies to `rpc/Cargo.toml`
- Implement `Arbitrary` for Soma domain types (or derive via `test-strategy`)
- Create the `protobuf_roundtrip_test!` macro in `rpc/src/utils/proptests.rs`

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `rpc/src/utils/proptests.rs` (create) | `crates/sui-rpc/src/proto/sui/rpc/v2/proptests.rs` |
| `rpc/src/utils/rpc_proto_conversions.rs` | `crates/sui-rpc/src/proto/` conversion impls |

---

## Priority 10: RPC Type Serialization Tests

**File to create**: `rpc/src/types/serialization_tests.rs`
**Sui equivalent**: `crates/sui-sdk-types/src/serialization_proptests.rs` (87 proptest roundtrip tests)

Sui generates two tests per type via `serialization_test!` macro:
- **Roundtrip**: BCS serialize → deserialize, JSON serialize → deserialize
- **Fuzz**: Arbitrary bytes → attempt BCS deserialization (must not panic)

### Tests to Implement (~20 tests covering key types)

| Test | Description |
|------|-------------|
| `test_address_roundtrip` | Address hex parsing → display → parse roundtrip |
| `test_address_from_str` | `Address::from_str("0x...")` with leading zeros |
| `test_digest_base58_roundtrip` | Digest → Base58 → parse → Digest |
| `test_transaction_bcs_roundtrip` | All TransactionKind variants BCS serialize/deserialize |
| `test_transaction_data_bcs_roundtrip` | Full TransactionData with gas, sender |
| `test_transaction_effects_bcs_roundtrip` | TransactionEffects BCS roundtrip |
| `test_object_bcs_roundtrip` | Object (Coin, Target, Submission, Challenge) BCS |
| `test_checkpoint_bcs_roundtrip` | CheckpointSummary BCS roundtrip |
| `test_checkpoint_contents_bcs_roundtrip` | CheckpointContents custom serialization |
| `test_execution_status_bcs_roundtrip` | All execution status variants |
| `test_transaction_fee_bcs_roundtrip` | TransactionFee struct BCS |
| `test_target_bcs_roundtrip` | Target with SomaTensor embedding |
| `test_challenge_bcs_roundtrip` | Challenge with all audit fields |
| `test_system_state_bcs_roundtrip` | SystemState (complex nested struct) |
| `test_user_signature_bcs_roundtrip` | UserSignature BCS |
| `test_multisig_bcs_roundtrip` | MultisigAggregatedSignature BCS |
| `test_bitmap_roundtrip` | Roaring bitmap serialize/deserialize |
| `test_hash_computation` | Blake2b256 hash correctness |
| `test_address_derivation` | Ed25519 public key → SomaAddress |
| `test_object_digest_computation` | Object contents → digest |

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `rpc/src/types/serialization_tests.rs` (create) | `crates/sui-sdk-types/src/serialization_proptests.rs` |
| `rpc/src/types/*.rs` | `crates/sui-sdk-types/src/*.rs` |

---

## Priority 11: RPC Proto Conversion Tests

**File to create**: `rpc/src/utils/conversion_tests.rs`
**Sui equivalent**: Covered by proptest roundtrips in Sui; Soma needs explicit tests for its largest untested file

The file `rpc/src/utils/rpc_proto_conversions.rs` is the largest file in the RPC crate (~27k tokens) with zero tests. It contains bidirectional conversions for every domain type to/from protobuf.

### Tests to Implement (~12 tests)

| Test | Description |
|------|-------------|
| `test_transaction_fee_proto_conversion` | TransactionFee ↔ proto roundtrip |
| `test_execution_status_success_conversion` | Success status ↔ proto |
| `test_execution_status_failure_conversion` | Each failure variant ↔ proto |
| `test_transaction_kind_transfer_coin` | TransferCoin ↔ proto |
| `test_transaction_kind_add_stake` | AddStake ↔ proto |
| `test_transaction_kind_submit_data` | SubmitData ↔ proto (Soma-specific) |
| `test_transaction_kind_commit_model` | CommitModel ↔ proto (Soma-specific) |
| `test_object_proto_conversion` | Object with different ObjectTypes ↔ proto |
| `test_system_state_proto_conversion` | SystemState full structure ↔ proto |
| `test_target_proto_conversion` | Target with embeddings, model_ids ↔ proto |
| `test_challenge_proto_conversion` | Challenge with audit data ↔ proto |
| `test_timestamp_conversion` | `timestamp_ms_to_proto` / `proto_to_timestamp_ms` roundtrip |

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `rpc/src/utils/rpc_proto_conversions.rs` | `crates/sui-rpc/src/proto/` conversion modules |
| `rpc/src/utils/types_conversions.rs` | Sui type conversion layer |

---

## Priority 12: RPC Field Mask Tests (expand existing)

**Files**: `rpc/src/utils/field/mod.rs`, `field_mask_tree.rs`, `field_mask_util.rs`
**Sui equivalent**: `crates/sui-rpc/src/field/` (7 tests — same count as Soma's 7)

Soma already has parity with Sui on field mask tests. Add edge case tests.

### Tests to Add (~5 tests)

| Test | Description |
|------|-------------|
| `test_field_mask_empty_path_filtered` | Empty paths ignored in FieldMask |
| `test_field_mask_tree_wildcard` | Wildcard `*` matches all subfields |
| `test_field_mask_tree_nested_overlap` | Adding "a" then "a.b" deduplicates to "a" |
| `test_field_mask_validate_nested_message` | Validate with real Soma proto message types |
| `test_field_mask_normalize_removes_redundant` | Normalize removes redundant sub-paths |

---

## Priority 13: RPC gRPC Handler Tests

**Files to test**: `rpc/src/api/grpc/ledger_service/`, `state_service/`, `transaction_execution_service/`
**Sui equivalent**: Sui tests handlers via integration tests; Soma should add unit tests with mock state

### Tests to Implement (~10 tests)

| Test | Handler | Description |
|------|---------|-------------|
| `test_get_service_info` | `get_service_info` | Returns chain_id, epoch, checkpoint_height |
| `test_get_object_found` | `get_object` | Object exists → correct response |
| `test_get_object_not_found` | `get_object` | Object missing → appropriate error |
| `test_get_object_with_field_mask` | `get_object` | Field mask limits response fields |
| `test_get_transaction_found` | `get_transaction` | Transaction found by digest |
| `test_get_balance_found` | `get_balance` | Balance returned for address with coins |
| `test_get_target_found` | `get_target` | Target returned by ID (Soma-specific) |
| `test_list_targets_status_filter` | `list_targets` | Filter by "open" status |
| `test_list_targets_pagination` | `list_targets` | Page token works correctly |
| `test_get_challenge_found` | `get_challenge` | Challenge returned by ID (Soma-specific) |

### Prerequisites

Requires `MockStateReader` test utility (see Infrastructure section).

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `rpc/src/api/grpc/state_service/get_target.rs` | N/A (Soma-original) |
| `rpc/src/api/grpc/ledger_service/get_object.rs` | `crates/sui-rpc/` handler equivalent |

---

## Priority 14: RPC Client Tests

**File to create**: `rpc/src/api/client_tests.rs`
**Sui equivalent**: `crates/integration-tests/` (6 integration tests)

Sui's integration tests use `SuiNetworkBuilder` to spawn a real Sui binary. Soma should add similar tests using `TestCluster`.

### Tests to Implement (~8 tests, in e2e-tests)

Note: These should be added as msim tests in `e2e-tests/tests/rpc_tests.rs` (already has 7 tests).

| Test | Description |
|------|-------------|
| `test_client_get_object` | Client fetches object by ID |
| `test_client_list_owned_objects` | Client lists objects for address |
| `test_client_get_balance` | Client gets SOMA balance |
| `test_client_execute_transaction` | Client executes transaction end-to-end |
| `test_client_subscribe_checkpoints` | Client subscribes to checkpoint stream |
| `test_client_get_target` | Client fetches target by ID |
| `test_client_list_targets` | Client lists targets with filters |
| `test_client_simulate_transaction` | Client simulates transaction without execution |

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `e2e-tests/tests/rpc_tests.rs` (expand) | `crates/integration-tests/tests/` |

---

## Priority 15: RPC Serde & Utilities Tests

**Files**: `rpc/src/utils/_serde/mod.rs`, `well_known_types.rs`, `rpc/src/proto/mod.rs`
**Sui equivalent**: `crates/sui-rpc/src/_serde/` (2 tests)

### Tests to Add (~5 tests)

| Test | Description |
|------|-------------|
| `test_well_known_timestamp_roundtrip` | Timestamp serde serialization/deserialization |
| `test_well_known_duration_roundtrip` | Duration serde roundtrip |
| `test_well_known_field_mask_roundtrip` | FieldMask serde roundtrip |
| `test_timestamp_ms_to_proto` | `timestamp_ms_to_proto(1234567890)` correctness |
| `test_proto_to_timestamp_ms` | `proto_to_timestamp_ms(proto)` correctness |

---

## Implementation Order

### Phase 1: CLI Unit Tests (Day 1-3)
1. **Create test infrastructure** — `cli/src/unit_tests/mod.rs`, utilities
2. **CLI pure function tests** — ~20 tests (Priority 1)
3. **CLI keytool tests** — ~10 tests (Priority 2)
4. **CLI response/display tests** — ~10 tests (Priority 3)

### Phase 2: RPC Type Tests (Day 4-7)
5. **RPC proptest infrastructure** — `proptest` dependency, `Arbitrary` impls, roundtrip macro
6. **RPC proto roundtrip tests** — ~15 proptests (Priority 9)
7. **RPC type serialization tests** — ~20 tests (Priority 10)
8. **RPC proto conversion tests** — ~12 tests (Priority 11)

### Phase 3: SDK Tests (Day 8-10)
9. **SDK client config tests** — ~8 tests (Priority 6)
10. **SDK error & builder tests** — ~6 tests (Priority 7)
11. **SDK wallet context tests** — ~8 tests (Priority 8)
12. **SDK proxy client expansion** — ~5 new tests

### Phase 4: RPC Handler & Client Tests (Day 11-13)
13. **RPC mock infrastructure** — `MockStateReader`
14. **RPC gRPC handler tests** — ~10 tests (Priority 13)
15. **RPC field mask expansion** — ~5 tests (Priority 12)
16. **RPC serde & utilities** — ~5 tests (Priority 15)

### Phase 5: Integration Tests (Day 14-16)
17. **CLI command integration tests** — ~18 simtest tests (Priority 4)
18. **CLI genesis ceremony test** — 1 test (Priority 5)
19. **RPC client integration tests** — ~8 tests via e2e-tests expansion (Priority 14)

### Phase 6: Attribution & Verification (Day 17)
20. **Add attribution headers** to all ~60 derived files across 3 crates
21. **Fix stale Sui references** in doc comments (8 files identified)
22. **Verify all tests pass**: `cargo test -p cli -p sdk -p rpc`
23. **Cross-check test count** against target (~195)

---

## Build & Run Commands

```bash
# Run CLI unit tests
cargo test -p cli

# Run SDK unit tests
cargo test -p sdk

# Run RPC unit tests (including proptests)
cargo test -p rpc

# Run specific test file
cargo test -p cli -- keytool_tests
cargo test -p sdk -- proxy_client
cargo test -p rpc -- proptests

# Run a single test
cargo test -p cli -- test_parse_hex_digest_32_valid

# Run RPC proptests with more cases
PROPTEST_CASES=1000 cargo test -p rpc -- proptests

# Build for msim (CLI integration tests)
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test cli_tests

# Run all RPC e2e tests (existing + new)
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test rpc_tests

# Check compilation only
PYO3_PYTHON=python3 cargo check -p cli -p sdk -p rpc

# Run with output
cargo test -p rpc -- --nocapture 2>&1 | head -100
```

---

## Summary of New Files to Create

| File | Crate | Tests | Priority |
|------|-------|-------|----------|
| `cli/src/unit_tests/mod.rs` | CLI | 0 (module registration) | Infrastructure |
| `cli/src/unit_tests/pure_function_tests.rs` | CLI | ~20 | P1 |
| `cli/src/unit_tests/keytool_tests.rs` | CLI | ~10 | P2 |
| `cli/src/unit_tests/response_tests.rs` | CLI | ~10 | P3 |
| `e2e-tests/tests/cli_tests.rs` | e2e-tests | ~18 | P4 |
| Inline in `genesis_ceremony/mod.rs` | CLI | 1 | P5 |
| Inline in `client_config.rs` | SDK | ~8 | P6 |
| Inline in `error.rs` + `transaction_builder.rs` | SDK | ~6 | P7 |
| Inline in `wallet_context.rs` | SDK | ~8 | P8 |
| `rpc/src/utils/proptests.rs` | RPC | ~15 | P9 |
| `rpc/src/types/serialization_tests.rs` | RPC | ~20 | P10 |
| `rpc/src/utils/conversion_tests.rs` | RPC | ~12 | P11 |
| Expand `rpc/src/utils/field/*.rs` | RPC | ~5 | P12 |
| `rpc/src/api/grpc/test_utils.rs` | RPC | 0 (infrastructure) | P13 |
| Inline handler tests | RPC | ~10 | P13 |
| Expand `e2e-tests/tests/rpc_tests.rs` | e2e-tests | ~8 | P14 |
| Expand `rpc/src/utils/_serde/` + proto | RPC | ~5 | P15 |
| Expand `sdk/src/proxy_client.rs` | SDK | ~5 | P8 |
| **Total new files** | | **~10** | |
| **Total new tests** | | **~163** | |

Combined with 32 existing tests, target is **~195 tests** across CLI, SDK, and RPC.

---

## Sui Cross-Reference URLs

### MystenLabs/sui (CLI, SDK)

| Category | Sui File Path |
|----------|--------------|
| CLI keytool tests | `crates/sui/src/unit_tests/keytool_tests.rs` |
| CLI integration tests | `crates/sui/tests/cli_tests.rs` |
| CLI keytool source | `crates/sui/src/keytool.rs` |
| CLI commands source | `crates/sui/src/sui_commands.rs` |
| CLI client commands | `crates/sui/src/client_commands.rs` |
| CLI validator commands | `crates/sui/src/validator_commands.rs` |
| CLI genesis ceremony | `crates/sui/src/genesis_ceremony.rs` |
| CLI genesis inspector | `crates/sui/src/genesis_inspector.rs` |
| SDK WalletContext | `crates/sui-sdk/src/wallet_context.rs` |
| SDK SuiClientConfig | `crates/sui-sdk/src/sui_client_config.rs` |
| SDK Error types | `crates/sui-sdk/src/error.rs` |

### MystenLabs/sui-rust-sdk (RPC)

| Category | sui-rust-sdk File Path |
|----------|----------------------|
| RPC protobuf roundtrip tests | `crates/sui-rpc/src/proto/sui/rpc/v2/proptests.rs` |
| RPC field mask util | `crates/sui-rpc/src/field/field_mask_util.rs` |
| RPC field mask tree | `crates/sui-rpc/src/field/field_mask_tree.rs` |
| RPC serde | `crates/sui-rpc/src/_serde/mod.rs` |
| RPC client | `crates/sui-rpc/src/client/mod.rs` |
| SDK types serialization tests | `crates/sui-sdk-types/src/serialization_proptests.rs` |
| SDK types address | `crates/sui-sdk-types/src/address.rs` |
| SDK types digest | `crates/sui-sdk-types/src/digest.rs` |
| SDK types hash | `crates/sui-sdk-types/src/hash.rs` |
| SDK types effects | `crates/sui-sdk-types/src/effects/mod.rs` |
| SDK types object | `crates/sui-sdk-types/src/object.rs` |
| SDK types transaction | `crates/sui-sdk-types/src/transaction/serialization.rs` |
| SDK types checkpoint | `crates/sui-sdk-types/src/checkpoint.rs` |
| Crypto ed25519 | `crates/sui-crypto/src/ed25519.rs` |
| Crypto bls12381 | `crates/sui-crypto/src/bls12381.rs` |
| Integration tests | `crates/integration-tests/tests/` |
| Transaction builder | `crates/sui-transaction-builder/src/builder.rs` |
