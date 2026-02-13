# Types Crate — Comprehensive Testing Plan

Testing plan for `types/src/` achieving high parity with Sui's `crates/sui-types/src/`. Covers file-by-file mapping, attribution requirements, and every test needed for parity plus Soma-specific coverage.

**Sui reference**: `MystenLabs/sui` — `crates/sui-types/src/`
**Soma crate**: `types/src/`

---

## Audit Notes (Feb 2026)

**Priority Ranking**: #3 of 7 plans — foundational for correctness. Types define serialization formats, crypto primitives, and data structures used by every other crate.

**Accuracy Correction**: The executive summary claims **124 existing test functions**, but codebase verification found **89 actual test functions**. The overcounting appears to come from counting `#[cfg(test)]` helper functions/fixtures as tests. The breakdown by file is accurate in the "Existing Test Coverage" section, but the total needs correction:
- External test files (5 files): 71 tests (verified)
- External test files (unit_tests/): 7 tests (verified)
- Inline test modules (9 files): The plan claims 46 inline tests, but actual count is ~11 lower. Discrepancy likely from counting test helper functions.

**Corrected total: ~89 existing tests, not 124.**

**Key Concerns**:
1. **Zero tests for core Sui-derived infrastructure** — `base.rs` (880 lines), `transaction.rs` (1988 lines), `crypto.rs` (1785 lines), `object.rs` (1132 lines), `effects/` (combined ~800 lines), `checkpoints.rs` (826 lines), `envelope.rs` (396 lines) all have zero tests. These are the serialization-critical files where BCS format bugs cause consensus divergence.
2. **Strong Soma-specific coverage** — submission (21 tests), target (17 tests), model (16 tests), rewards (9 tests), delegation (8 tests) are well-tested. This is the bright spot.
3. **Genesis builder (706 lines, 0 tests)** — genesis correctness is a one-time but irreversible operation. A genesis bug requires chain restart.
4. **Transaction serialization for all 31 variants** — BCS roundtrip tests for every `TransactionKind` variant should be the highest priority within this plan, as serialization bugs cause consensus splits.
5. **Missing: property-based testing** — proptest/quickcheck for serialization roundtrips would catch edge cases that manual tests miss.

**Estimated Effort**: ~14 engineering days as planned. Recommend prioritizing transaction and effects BCS roundtrip tests first.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [File-by-File Cross-Reference](#file-by-file-cross-reference)
3. [Attribution Requirements](#attribution-requirements)
4. [Existing Test Coverage](#existing-test-coverage)
5. [Priority 1: Base Types Tests](#priority-1-base-types-tests)
6. [Priority 2: Transaction & Message Tests](#priority-2-transaction--message-tests)
7. [Priority 3: Crypto & Intent Tests](#priority-3-crypto--intent-tests)
8. [Priority 4: Effects Tests](#priority-4-effects-tests)
9. [Priority 5: Object Tests](#priority-5-object-tests)
10. [Priority 6: Checkpoint Tests](#priority-6-checkpoint-tests)
11. [Priority 7: Envelope & Signature Verification Tests](#priority-7-envelope--signature-verification-tests)
12. [Priority 8: Committee Tests](#priority-8-committee-tests)
13. [Priority 9: MultiSig Tests (Extend)](#priority-9-multisig-tests-extend)
14. [Priority 10: Genesis Builder Tests](#priority-10-genesis-builder-tests)
15. [Priority 11: Temporary Store Tests](#priority-11-temporary-store-tests)
16. [Priority 12: Storage Layer Tests](#priority-12-storage-layer-tests)
17. [Priority 13: Config Tests](#priority-13-config-tests)
18. [Priority 14: Consensus Types Tests](#priority-14-consensus-types-tests)
19. [Priority 15: Error & Serde Tests](#priority-15-error--serde-tests)
20. [Soma-Specific Coverage (Already Exists — Verify)](#soma-specific-coverage-already-exists--verify)
21. [Implementation Order](#implementation-order)
22. [Build & Run Commands](#build--run-commands)

---

## Executive Summary

### Current State
- **~99 Rust files** across the types crate (~36,500 lines)
- **124 existing test functions** across 15 files
- Strong Soma-specific coverage: submission (23 tests), target (21 tests), model (16 tests), rewards (9 tests), delegation (8 tests)
- **Zero tests** for core infrastructure files adopted from Sui: `base.rs`, `transaction.rs`, `crypto.rs`, `object.rs`, `effects/`, `checkpoints.rs`, `envelope.rs`, `intent.rs`, `committee.rs`, `genesis_builder.rs`, `temporary_store.rs`, `config/`, `storage/`
- Partial multisig coverage: 7 of Sui's 13 tests implemented

### Target State
- **~300+ unit tests** — matching Sui's ~275 relevant tests + Soma-specific tests
- New test files for all core infrastructure modules
- Complete attribution headers on all derived files

### Test Count Summary

| Category | Sui Tests | Soma Existing | Gap |
|----------|-----------|---------------|-----|
| Base types (ObjectID, Address, Digest, serde) | 31 | 0 | ~25 |
| Transaction/Messages (signing, certs, quorum) | 24 | 0 | ~20 |
| Crypto (keypairs, proof-of-possession, serde) | 11 | 0 | ~8 |
| Effects (written objects, status, object_change) | 20 | 0 | ~15 |
| Object (digest, coin value, serde) | 3 | 0 | 3 |
| Checkpoint (signed, certified, digest) | 4 | 0 | 4 |
| Committee (shuffle_by_weight) | 1 | 0 | 1 |
| MultiSig | 13 | 7 | 6 |
| Intent | 2 | 0 | 2 |
| Balance change | 16 | 0 | ~8 |
| Envelope | 0 | 0 | ~5 |
| Genesis builder | 0 | 0 | ~10 |
| Temporary store | 0 | 0 | ~8 |
| Storage layer | 0 | 0 | ~5 |
| Config | 0 | 0 | ~6 |
| Consensus types | 0 | 0 | ~5 |
| Error & serde | 0 | 0 | ~4 |
| Soma-specific (existing — verify) | N/A | 117 | 0 |
| **Total** | **~125 relevant** | **124** | **~135** |

---

## File-by-File Cross-Reference

### Legend
- **Heavy** = Direct port/fork from Sui, needs full attribution
- **Moderate** = Significant shared patterns/structure, needs attribution
- **Light** = Minor influence, attribution recommended
- **Soma-only** = Original Soma code, no attribution needed
- **N/A** = Sui file has no Soma counterpart (Move VM, zkLogin, etc.)

### Core Identity & Base Types

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `base.rs` | `base_types.rs` | Heavy | 0 | 880 |
| `digests.rs` | `digests.rs` | Heavy | 0 | 1180 |
| `object.rs` | `object.rs` + `object/` | Heavy | 0 | 1132 |
| `error.rs` | `error.rs` | Heavy | 0 | 944 |
| `balance_change.rs` | `balance_change.rs` | Heavy | 0 | 103 |
| `peer_id.rs` | — | Light | 0 | 130 |
| `validator_info.rs` | — | Moderate | 0 | 167 |

### Transactions & Messages

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `transaction.rs` | `transaction.rs` | Moderate | 0 | 1988 |
| `messages_grpc.rs` | `messages_grpc.rs` | Heavy | 0 | 872 |
| `transaction_outputs.rs` | — | Moderate | 0 | 137 |
| `transaction_executor.rs` | `transaction_executor.rs` | Heavy | 0 | 51 |
| `tx_fee.rs` | — | Soma-only | 0 | 51 |
| `quorum_driver.rs` | `transaction_driver_types.rs` | Heavy | 0 | 198 |
| `execution.rs` | `execution.rs` | Heavy | 0 | 123 |

### Crypto & Authentication

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `crypto.rs` | `crypto.rs` + `signature.rs` | Heavy | 0 | 1785 |
| `multisig.rs` | `multisig.rs` | Heavy | 7 (external) | 354 |
| `intent.rs` | — (Soma's own) | Moderate | 0 | 148 |
| `signature_verification.rs` | `signature_verification.rs` | Heavy | 0 | 70 |
| `envelope.rs` | `message_envelope.rs` | Heavy | 0 | 396 |
| `serde.rs` | `sui_serde.rs` | Heavy | 0 | 103 |

### Checkpoints & Finality

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `checkpoints.rs` | `messages_checkpoint.rs` | Heavy | 0 | 826 |
| `full_checkpoint_content.rs` | `full_checkpoint_content.rs` | Heavy | 0 | 332 |
| `finality.rs` | — | Soma-only | 0 | 114 |

### Committee & Governance

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `committee.rs` | `committee.rs` | Heavy | 0 | 718 |
| `supported_protocol_versions.rs` | `supported_protocol_versions.rs` | Heavy | 0 | 85 |

### System State

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `system_state/mod.rs` | `sui_system_state/mod.rs` + `*_inner_v1.rs` | Moderate | 78 (5 ext. files) | 1422 |
| `system_state/validator.rs` | `sui_system_state/` + `governance.rs` | Moderate | 0 (inline infra only) | 1215 |
| `system_state/staking.rs` | `governance.rs` | Moderate | 0 (inline infra only) | 358 |
| `system_state/model_registry.rs` | — | Soma-only | 0 | 58 |
| `system_state/target_state.rs` | — | Soma-only | 6 (inline) | 231 |
| `system_state/emission.rs` | — | Soma-only | 0 | 39 |
| `system_state/epoch_start.rs` | `epoch_start_sui_system_state.rs` | Moderate | 0 | 239 |

### Config

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `config/mod.rs` | `sui-config/` | Heavy | 0 | 191 |
| `config/genesis_config.rs` | `sui-swarm-config/genesis_config.rs` | Heavy | 0 | 487 |
| `config/node_config.rs` | `sui-config/node.rs` | Heavy | 0 | 1213 |
| `config/network_config.rs` | `sui-config/` | Heavy | 0 | 393 |
| `config/p2p_config.rs` | `sui-config/p2p.rs` | Heavy | 0 | 104 |
| `config/state_sync_config.rs` | `sui-config/` | Heavy | 0 | 168 |
| `config/rpc_config.rs` | — | Moderate | 0 | 121 |
| `config/object_store_config.rs` | — | Moderate | 0 | 239 |
| `config/certificate_deny_config.rs` | — | Heavy | 0 | 56 |
| `config/transaction_deny_config.rs` | — | Heavy | 0 | 103 |
| `config/local_ip_utils.rs` | — | Moderate | 0 | 123 |
| `config/validator_client_monitor_config.rs` | — | Moderate | 0 | 161 |

### Storage

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `storage/mod.rs` | `storage/mod.rs` | Heavy | 0 | 456 |
| `storage/read_store.rs` | `storage/read_store.rs` | Heavy | 0 | 607 |
| `storage/write_store.rs` | `storage/write_store.rs` | Heavy | 0 | 115 |
| `storage/object_store.rs` | `storage/object_store_trait.rs` | Heavy | 0 | 113 |
| `storage/shared_in_memory_store.rs` | `storage/shared_in_memory_store.rs` | Heavy | 0 | 535 |
| `storage/storage_error.rs` | `storage/error.rs` | Heavy | 0 | 64 |
| `storage/committee_store.rs` | — | Moderate | 0 | 105 |
| `storage/write_path_pending_tx_log.rs` | — | Moderate | 0 | 72 |
| `storage/consensus/mod.rs` | — | Moderate | 0 | 102 |
| `storage/consensus/mem_store.rs` | — | Moderate | 0 | 212 |
| `storage/consensus/rocksdb_store.rs` | — | Moderate | 0 | 283 |

### Consensus Types

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `consensus/mod.rs` | `messages_consensus.rs` | Heavy | 0 | 368 |
| `consensus/block.rs` | `consensus/core/src/block.rs` types | Heavy | test infra only | 646 |
| `consensus/commit.rs` | `consensus/core/src/commit.rs` types | Heavy | test infra only | 655 |
| `consensus/context.rs` | `consensus/core/src/context.rs` types | Heavy | 0 | 162 |
| `consensus/leader_scoring.rs` | `consensus/core/src/leader_scoring.rs` | Heavy | 0 | 177 |
| `consensus/stake_aggregator.rs` | — | Moderate | test infra only | 85 |
| `consensus/validator_set.rs` | — | Moderate | 0 | 54 |

### Sync

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `sync/mod.rs` | `sui-network/` types | Heavy | 0 | 182 |
| `sync/active_peers.rs` | — | Moderate | 0 | 112 |
| `sync/channel_manager.rs` | — | Moderate | 0 | 499 |

### TLS

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `tls/mod.rs` | `sui-tls/` | Heavy | 5 (inline) | 329 |
| `tls/acceptor.rs` | `sui-tls/` | Heavy | 0 | 84 |
| `tls/certgen.rs` | `sui-tls/` | Heavy | 0 | 57 |
| `tls/verifier.rs` | `sui-tls/` | Heavy | 0 | 313 |

### Other Utilities

| Soma File | Sui File | Derivation | Existing Tests | Lines |
|-----------|----------|------------|----------------|-------|
| `genesis_builder.rs` | `sui-genesis-builder/src/lib.rs` | Heavy | 0 | 706 |
| `genesis.rs` | — | Heavy | 0 | 246 |
| `temporary_store.rs` | `inner_temporary_store.rs` | Heavy | 0 | 707 |
| `mutex_table.rs` | — | Moderate | 0 | 263 |
| `traffic_control.rs` | `traffic_control.rs` | Heavy | 0 | 56 |
| `parameters.rs` | — | Soma-only | 0 | 335 |
| `grpc_timeout.rs` | — | Moderate | 10 (inline) | 271 |
| `multiaddr.rs` | — | Moderate | 6 (inline) | 430 |
| `metadata.rs` | — | Soma-only | 0 | 87 |
| `client.rs` | `client.rs` | Heavy | 0 | 378 |
| `checksum.rs` | — | Soma-only | 0 | 131 |

### Soma-Only Files (No Sui Equivalent)

| Soma File | Existing Tests | Lines |
|-----------|----------------|-------|
| `target.rs` | 4 (inline) | 484 |
| `submission.rs` | 2 (inline) | 162 |
| `challenge.rs` | 3 (inline) | 345 |
| `model.rs` | 0 | 120 |
| `model_selection.rs` | 10 (inline) | 419 |
| `tensor.rs` | 0 (re-export) | 6 |
| `finality.rs` | 0 | 114 |
| `system_state/model_registry.rs` | 0 | 58 |
| `system_state/target_state.rs` | 6 (inline) | 231 |
| `system_state/emission.rs` | 0 | 39 |
| `tx_fee.rs` | 0 | 51 |

### Sui Files With No Soma Counterpart

| Sui File | Reason Not Applicable |
|----------|----------------------|
| `programmable_transaction_builder.rs` | Soma uses native tx types, not PTBs |
| `move_package.rs` | No Move VM |
| `gas.rs`, `gas_coin.rs`, `gas_model/` | Soma uses flat-fee model, not metered gas |
| `zk_login_authenticator.rs`, `zk_login_util.rs` | Sui-specific auth |
| `passkey_authenticator.rs` | Sui-specific auth |
| `nitro_attestation.rs` | Sui-specific TEE |
| `deny_list_v1.rs`, `deny_list_v2.rs` | Sui-specific deny lists |
| `bridge.rs` | Sui bridge |
| `display.rs`, `display_registry.rs` | Sui Display objects |
| `coin.rs`, `coin_registry.rs`, `coin_reservation.rs` | Move coin objects |
| `balance.rs`, `funds_accumulator.rs` | Move Balance type |
| `accumulator_event.rs`, `accumulator_metadata.rs`, `accumulator_root.rs` | Sui accumulators |
| `address_alias.rs` | Sui address aliases |
| `authenticator_state.rs`, `randomness_state.rs`, `clock.rs` | Sui system objects |
| `dynamic_field/` | Move dynamic fields |
| `id.rs`, `config.rs`, `derived_object.rs`, `transfer.rs` | Move-specific |
| `collection_types.rs` | Move VecMap/VecSet |
| `multisig_legacy.rs` | Legacy compat |
| `versioned.rs` | Sui versioning |
| `test_checkpoint_data_builder.rs` | Sui test infrastructure |
| `ptb_trace.rs` | PTB tracing |

---

## Attribution Requirements

All files marked **Heavy** or **Moderate** derivation above need the following header:

```rust
// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-types/src/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
```

### Files Requiring Attribution (67 files)

**Core Identity & Base Types (6):**
- `base.rs`, `digests.rs`, `object.rs`, `error.rs`, `balance_change.rs`, `validator_info.rs`

**Transactions & Messages (6):**
- `transaction.rs`, `messages_grpc.rs`, `transaction_outputs.rs`, `transaction_executor.rs`, `quorum_driver.rs`, `execution.rs`

**Crypto & Authentication (5):**
- `crypto.rs`, `multisig.rs`, `signature_verification.rs`, `envelope.rs`, `serde.rs`

**Checkpoints (2):**
- `checkpoints.rs`, `full_checkpoint_content.rs`

**Committee (2):**
- `committee.rs`, `supported_protocol_versions.rs`

**System State (4):**
- `system_state/mod.rs`, `system_state/validator.rs`, `system_state/staking.rs`, `system_state/epoch_start.rs`

**Config (10):**
- `config/mod.rs`, `config/genesis_config.rs`, `config/node_config.rs`, `config/network_config.rs`
- `config/p2p_config.rs`, `config/state_sync_config.rs`, `config/certificate_deny_config.rs`
- `config/transaction_deny_config.rs`, `config/local_ip_utils.rs`, `config/validator_client_monitor_config.rs`

**Storage (8):**
- `storage/mod.rs`, `storage/read_store.rs`, `storage/write_store.rs`, `storage/object_store.rs`
- `storage/shared_in_memory_store.rs`, `storage/storage_error.rs`, `storage/committee_store.rs`
- `storage/write_path_pending_tx_log.rs`

**Consensus Types (5):**
- `consensus/mod.rs`, `consensus/block.rs`, `consensus/commit.rs`, `consensus/context.rs`, `consensus/leader_scoring.rs`

**Sync (2):**
- `sync/mod.rs`, `sync/active_peers.rs`

**TLS (4):**
- `tls/mod.rs`, `tls/acceptor.rs`, `tls/certgen.rs`, `tls/verifier.rs`

**Other (9):**
- `genesis_builder.rs`, `genesis.rs`, `temporary_store.rs`, `traffic_control.rs`
- `client.rs`, `intent.rs`, `grpc_timeout.rs`, `multiaddr.rs`, `mutex_table.rs`

**Test Files (2):**
- `unit_tests/multisig_tests.rs`, `unit_tests/utils.rs`

### Files NOT Requiring Attribution (Soma-only — 15 files)
- `target.rs`, `submission.rs`, `challenge.rs`, `model.rs`, `model_selection.rs`
- `tensor.rs`, `finality.rs`, `tx_fee.rs`, `parameters.rs`, `metadata.rs`, `checksum.rs`
- `system_state/model_registry.rs`, `system_state/target_state.rs`, `system_state/emission.rs`
- `peer_id.rs`

---

## Existing Test Coverage

### Summary: 124 existing tests across 15 files

#### External Test Files (system_state/unit_tests/ — 5 files, 71 tests)

| File | Tests | Coverage |
|------|-------|---------|
| `submission_tests.rs` | 21 | Submission lifecycle, bond calc, fraud quorum, tally, serialization, edge cases |
| `target_tests.rs` | 17 | Target generation, difficulty adjustment, EMA, status transitions, model selection |
| `model_tests.rs` | 16 | Commit-reveal, update, deactivation, commission, report quorum, slash, errors |
| `rewards_distribution_tests.rs` | 9 | Validator rewards, stake subsidy, commission, slashing, multi-epoch |
| `delegation_tests.rs` | 8 | Staking pool flows, exchange rates, activation/deactivation, preactive validators |

#### External Test Files (unit_tests/ — 1 file, 7 tests)

| File | Tests | Coverage |
|------|-------|---------|
| `multisig_tests.rs` | 7 | Combine sigs, serde roundtrip, MultiSigPublicKey validation, address, max sig, pk/indices |

#### Inline Test Modules (9 files, 46 tests)

| File | Tests | Coverage |
|------|-------|---------|
| `grpc_timeout.rs` | 10 | All timeout units, missing header, invalid input |
| `model_selection.rs` | 10 | KNN selection, stake weighting, voting power, batch select |
| `system_state/target_state.rs` | 6 | Default, new, counters, EMA bootstrap/update, no-targets |
| `tls/mod.rs` | 5 | AllowAll, ServerCertVerifier, HashSetAllow, invalid name, axum TLS |
| `target.rs` | 4 | Deterministic embedding, dimension, seed, status methods |
| `challenge.rs` | 3 | Creation, reports lifecycle, status transitions |
| `multiaddr.rs` | 6 | Socket addr conversion, TCP validity, hostname/port, zero/localhost IP |
| `submission.rs` | 2 | Manifest size, submission creation |
| `peer_id.rs` | 0 | (test infrastructure exists, no test functions) |

---

## Priority 1: Base Types Tests

**Soma file to create:** `types/src/unit_tests/base_types_tests.rs`
**Sui equivalent:** `crates/sui-types/src/unit_tests/base_types_tests.rs` (31 tests)

Sui has 31 tests for base types. After filtering out Move-specific tests (`test_move_object_size_for_gas_metering`, `test_move_package_size_for_gas_metering`, `move_object_type_consistency`), ~25 are directly applicable.

### Tests to Implement

| # | Test | Description | Sui Source |
|---|------|-------------|-----------|
| 1 | `test_bcs_owner_enum` | BCS serialization of `Owner` enum variants (AddressOwner, Shared, Immutable) | `test_bcs_enum` |
| 2 | `test_signatures` | Signature creation, verify with correct/wrong address/message | `test_signatures` |
| 3 | `test_signatures_serde` | Signature BCS serialize/deserialize roundtrip | `test_signatures_serde` |
| 4 | `test_max_sequence_number` | `Version::MAX` boundary check | `test_max_sequence_number` |
| 5 | `test_lamport_increment_version` | Lamport version incrementing | `test_lamport_increment_version` |
| 6 | `test_object_id_display` | ObjectID debug display format | `test_object_id_display` |
| 7 | `test_object_id_str_lossless` | ObjectID `short_str_lossless()` formatting | `test_object_id_str_lossless` |
| 8 | `test_object_id_from_hex_literal` | ObjectID from "0x..." literal | `test_object_id_from_hex_literal` |
| 9 | `test_object_id_ref` | ObjectID `as_ref()` to byte slice | `test_object_id_ref` |
| 10 | `test_object_id_from_proto_invalid_length` | ObjectID from invalid bytes | `test_object_id_from_proto_invalid_length` |
| 11 | `test_object_id_serde_json` | ObjectID JSON serialization format | `test_object_id_serde_json` |
| 12 | `test_object_id_serde_not_human_readable` | ObjectID BCS serialization | `test_object_id_serde_not_human_readable` |
| 13 | `test_object_id_serde_with_expected_value` | ObjectID pinned serialization output | `test_object_id_serde_with_expected_value` |
| 14 | `test_object_id_zero_padding` | ObjectID zero-padding from various formats | `test_object_id_zero_padding` |
| 15 | `test_object_id_from_empty_string` | ObjectID from empty string error | `test_object_id_from_empty_string` |
| 16 | `test_address_display` | SomaAddress debug display format | `test_address_display` |
| 17 | `test_address_serde_not_human_readable` | SomaAddress BCS serialization | `test_address_serde_not_human_readable` |
| 18 | `test_address_serde_human_readable` | SomaAddress JSON serialization | `test_address_serde_human_readable` |
| 19 | `test_address_serde_with_expected_value` | SomaAddress pinned serialization | `test_address_serde_with_expected_value` |
| 20 | `test_address_backwards_compatibility` | Pinned address derivation backwards-compat | `test_address_backwards_compatibility` |
| 21 | `test_transaction_digest_serde_not_human_readable` | TransactionDigest BCS | `test_transaction_digest_serde_*` |
| 22 | `test_transaction_digest_serde_human_readable` | TransactionDigest JSON (Base58) | `test_transaction_digest_serde_*` |
| 23 | `test_authority_signature_serde_not_human_readable` | AuthoritySignature BCS | `test_authority_signature_serde_*` |
| 24 | `test_authority_signature_serde_human_readable` | AuthoritySignature JSON (Base64) | `test_authority_signature_serde_*` |
| 25 | `test_size_one_vec_is_transparent` | `SizeOneVec` BCS transparency (Soma has this type in `base.rs`) | inline in `base_types.rs` |

### Wire to lib.rs

Add to `types/src/base.rs` (or `lib.rs` via `#[path]`):
```rust
#[cfg(test)]
#[path = "unit_tests/base_types_tests.rs"]
mod base_types_tests;
```

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/base.rs` | `crates/sui-types/src/base_types.rs` |
| `types/src/digests.rs` | `crates/sui-types/src/digests.rs` |
| `types/src/unit_tests/utils.rs` | `crates/sui-types/src/unit_tests/utils.rs` |

---

## Priority 2: Transaction & Message Tests

**Soma file to create:** `types/src/unit_tests/transaction_tests.rs`
**Sui equivalent:** `crates/sui-types/src/unit_tests/messages_tests.rs` (24 tests)

Sui's `messages_tests.rs` covers transaction signing, certificate assembly, quorum validation, digest caching, and system transaction properties. Many tests are directly applicable after adapting `SuiTransaction` → Soma's 31 `TransactionKind` variants.

### Tests to Implement

| # | Test | Description | Sui Source |
|---|------|-------------|-----------|
| 1 | `test_signed_values` | SignedTransaction verification with correct/wrong key | `test_signed_values` |
| 2 | `test_certificates` | CertifiedTransaction quorum logic (need 2f+1 sigs) | `test_certificates` |
| 3 | `test_new_with_signatures` | `AuthorityStrongQuorumSignInfo` creation and verification | `test_new_with_signatures` |
| 4 | `test_handle_reject_malicious_signature` | Reject malicious aggregate signature | `test_handle_reject_malicious_signature` |
| 5 | `test_auth_sig_wrong_epoch` | Epoch mismatch in authority signature rejected | `test_auth_sig_commit_to_wrong_epoch_id_fail` |
| 6 | `test_bitmap_out_of_range` | Bitmap out of range insertion | `test_bitmap_out_of_range` |
| 7 | `test_reject_extra_public_key` | Extra public key in quorum | `test_reject_extra_public_key` |
| 8 | `test_reject_reuse_signatures` | Duplicate signature detection | `test_reject_reuse_signatures` |
| 9 | `test_empty_bitmap` | Empty bitmap rejection | `test_empty_bitmap` |
| 10 | `test_digest_caching` | Digest caching and invalidation | `test_digest_caching` |
| 11 | `test_user_signature_committed_in_transactions` | User sig affects tx hash but not digest | `test_user_signature_committed_in_transactions` |
| 12 | `test_change_epoch_transaction` | ChangeEpoch system tx properties (is_system_tx, etc.) | `test_change_epoch_transaction` |
| 13 | `test_consensus_commit_prologue_transaction` | ConsensusCommitPrologue tx properties | `test_consensus_commit_prologue_transaction` |
| 14 | `test_certificate_digest` | Certificate digest changes on mutation | `test_certificate_digest` |
| 15 | `test_all_31_tx_kinds_bcs_roundtrip` | BCS serialize/deserialize roundtrip for all 31 TransactionKind variants | Soma-specific |
| 16 | `test_transaction_digest_determinism` | Same inputs produce same digest | Soma-specific |
| 17 | `test_transaction_kind_classification` | `is_system_tx()`, `requires_system_state()`, `is_submission_tx()` for each kind | Soma-specific |
| 18 | `test_gas_data_validation` | Empty gas, duplicate gas, gas wrong owner | Sui `messages_tests` patterns |
| 19 | `test_user_cannot_send_system_transactions` | User cannot send Genesis/ChangeEpoch/CCP | Sui `transaction_tests` |
| 20 | `test_sponsored_transaction` | Gas sponsor != sender validation | `test_sponsored_transaction_message` (adapted) |

### Wire to lib.rs

Add to `types/src/transaction.rs`:
```rust
#[cfg(test)]
#[path = "unit_tests/transaction_tests.rs"]
mod transaction_tests;
```

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/transaction.rs` | `crates/sui-types/src/transaction.rs` |
| `types/src/unit_tests/transaction_tests.rs` (create) | `crates/sui-types/src/unit_tests/messages_tests.rs` |
| `types/src/crypto.rs` | `crates/sui-types/src/crypto.rs` |
| `types/src/envelope.rs` | `crates/sui-types/src/message_envelope.rs` |

---

## Priority 3: Crypto & Intent Tests

**Soma file to create:** `types/src/unit_tests/crypto_tests.rs`
**Sui equivalent:** `crates/sui-types/src/unit_tests/crypto_tests.rs` (9 tests) + `intent_tests.rs` (2 tests)

### Tests to Implement

| # | Test | Description | Sui Source |
|---|------|-------------|-----------|
| 1 | `test_serde_keypair` | SomaKeyPair encode/decode roundtrip | `serde_keypair` |
| 2 | `test_serde_pubkey` | PublicKey JSON serialization | `serde_pubkey` |
| 3 | `test_serde_authority_quorum_sign_info` | AuthorityQuorumSignInfo JSON + BCS roundtrip | `serde_round_trip_authority_quorum_sign_info` |
| 4 | `test_public_key_equality` | PublicKey reflexivity, cross-scheme, different key | `public_key_equality` |
| 5 | `test_proof_of_possession` | Proof of possession generation and verification | `test_proof_of_possession` |
| 6 | `test_get_key_pair_from_bytes` | Key pair from random bytes no panic | `test_get_key_pair_from_bytes` |
| 7 | `test_from_signable_bytes` | Deserialize from random bytes no panic | `test_from_signable_bytes` |
| 8 | `test_personal_message_intent` | PersonalMessage intent BCS, domain separation, sign/verify | `test_personal_message_intent` |
| 9 | `test_authority_signature_intent` | Authority signature intent BCS, domain separation | `test_authority_signature_intent` |

### Wire

Add to `types/src/crypto.rs`:
```rust
#[cfg(test)]
#[path = "unit_tests/crypto_tests.rs"]
mod crypto_tests;
```

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/crypto.rs` | `crates/sui-types/src/crypto.rs` |
| `types/src/intent.rs` | `crates/sui-types/src/crypto.rs` (intent tests are loaded from crypto.rs) |
| `types/src/unit_tests/crypto_tests.rs` (create) | `crates/sui-types/src/unit_tests/crypto_tests.rs` + `intent_tests.rs` |

---

## Priority 4: Effects Tests

**Soma file to create:** `types/src/unit_tests/effects_tests.rs`
**Sui equivalent:** `crates/sui-types/src/unit_tests/effects_tests.rs` (10 tests) + `effects/object_change.rs` inline (10 tests)

### Tests to Implement — TransactionEffects

| # | Test | Description | Sui Source |
|---|------|-------------|-----------|
| 1 | `test_written_with_created_objects` | `effects.written()` includes created objects | `effects_tests.rs` |
| 2 | `test_written_with_mutated_objects` | `effects.written()` includes mutated objects | `effects_tests.rs` |
| 3 | `test_written_with_deleted_objects` | `effects.written()` includes deleted with DELETED digest | `effects_tests.rs` |
| 4 | `test_written_with_combination_of_all_types` | All 3+ change types combined | `effects_tests.rs` |
| 5 | `test_written_empty_when_no_changes` | Only gas when no changes | `effects_tests.rs` |
| 6 | `test_written_version_numbers` | All objects have same lamport version | `effects_tests.rs` |
| 7 | `test_effects_bcs_roundtrip` | BCS serialize/deserialize TransactionEffects | Soma-specific |
| 8 | `test_effects_digest_determinism` | Same effects → same digest | Soma-specific |
| 9 | `test_execution_status_variants` | Success, Failure with each ExecutionFailureStatus variant | Soma-specific |

### Tests to Implement — ObjectChange

| # | Test | Description | Sui Source |
|---|------|-------------|-----------|
| 10 | `test_object_in_out_variants` | ObjectIn/ObjectOut enum coverage | `effects/object_change.rs` inline |
| 11 | `test_id_operation_variants` | IDOperation enum coverage (Created, Deleted, None) | `effects/object_change.rs` inline |
| 12 | `test_effects_object_change_bcs` | EffectsObjectChange BCS roundtrip | Soma-specific |

### Soma-Specific Effects Tests

| # | Test | Description |
|---|------|-------------|
| 13 | `test_execution_failure_status_completeness` | Every ExecutionFailureStatus variant has a display string | Soma-specific |
| 14 | `test_effects_gas_summary` | GasCostSummary fields (computation, storage, rebate) | Soma-specific |
| 15 | `test_effects_shared_objects` | Shared object version tracking in effects | Soma-specific |

### Wire

Add to `types/src/effects/mod.rs`:
```rust
#[cfg(test)]
#[path = "../unit_tests/effects_tests.rs"]
mod effects_tests;
```

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/effects/mod.rs` | `crates/sui-types/src/effects/mod.rs` |
| `types/src/effects/object_change.rs` | `crates/sui-types/src/effects/object_change.rs` |
| `types/src/unit_tests/effects_tests.rs` (create) | `crates/sui-types/src/unit_tests/effects_tests.rs` |

---

## Priority 5: Object Tests

**Soma file to create:** `types/src/unit_tests/object_tests.rs`
**Sui equivalent:** `crates/sui-types/src/object.rs` inline (3 tests)

### Tests to Implement

| # | Test | Description | Sui Source |
|---|------|-------------|-----------|
| 1 | `test_object_digest_and_serialized_format` | Object digest stability, serialized format pinning | `test_object_digest_and_serialized_format` |
| 2 | `test_get_coin_value` | Get coin value from Object | `test_get_coin_value_unsafe` (adapted) |
| 3 | `test_set_coin_value` | Set coin value on Object | `test_set_coin_value_unsafe` (adapted) |
| 4 | `test_object_bcs_roundtrip` | BCS serialize/deserialize Object with each ObjectType | Soma-specific |
| 5 | `test_object_id_deterministic` | ObjectID generation is deterministic | Soma-specific |
| 6 | `test_owner_variants` | AddressOwner, Shared, Immutable serialization/comparison | Soma-specific |
| 7 | `test_version_ordering` | Version comparison and ordering | Soma-specific |
| 8 | `test_object_ref_tuple` | ObjectRef = (ObjectID, Version, Digest) | Soma-specific |
| 9 | `test_object_type_classification` | `ObjectType` variant checks (SystemState, Coin, Target, etc.) | Soma-specific |
| 10 | `test_shared_object_initial_version` | Shared objects store initial_shared_version correctly | Soma-specific |

### Wire

Add to `types/src/object.rs`:
```rust
#[cfg(test)]
#[path = "unit_tests/object_tests.rs"]
mod object_tests;
```

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/object.rs` | `crates/sui-types/src/object.rs` |

---

## Priority 6: Checkpoint Tests

**Soma file to create:** `types/src/unit_tests/checkpoint_tests.rs`
**Sui equivalent:** `crates/sui-types/src/messages_checkpoint.rs` inline (4 tests)

### Tests to Implement

| # | Test | Description | Sui Source |
|---|------|-------------|-----------|
| 1 | `test_signed_checkpoint` | Create and verify SignedCheckpointSummary | `test_signed_checkpoint` |
| 2 | `test_certified_checkpoint` | Create and verify CertifiedCheckpointSummary with quorum | `test_certified_checkpoint` |
| 3 | `test_checkpoint_summary_different_consensus_digest` | Different consensus digest → different checkpoint | `test_checkpoint_summary_with_different_consensus_digest` |
| 4 | `test_checkpoint_bcs_roundtrip` | CheckpointSummary BCS serialize/deserialize | Soma-specific |
| 5 | `test_checkpoint_sequence_ordering` | CheckpointSequenceNumber ordering | Soma-specific |
| 6 | `test_checkpoint_inclusion_proof` | Merkle inclusion proof verification | Soma-specific |
| 7 | `test_full_checkpoint_content` | FullCheckpointContent construction and serialization | Soma-specific |

### Wire

Add to `types/src/checkpoints.rs`:
```rust
#[cfg(test)]
#[path = "unit_tests/checkpoint_tests.rs"]
mod checkpoint_tests;
```

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/checkpoints.rs` | `crates/sui-types/src/messages_checkpoint.rs` |
| `types/src/full_checkpoint_content.rs` | `crates/sui-types/src/full_checkpoint_content.rs` |
| `types/src/finality.rs` | — (Soma-only) |

---

## Priority 7: Envelope & Signature Verification Tests

**Soma file to create:** `types/src/unit_tests/envelope_tests.rs`
**Sui equivalent:** `crates/sui-types/src/message_envelope.rs` (no inline tests, but heavily used)

Sui's `message_envelope.rs` has no dedicated tests, but the Envelope pattern is tested indirectly through `messages_tests.rs` (certificates, signed transactions). We need dedicated coverage for Soma's `envelope.rs`.

### Tests to Implement

| # | Test | Description |
|---|------|-------------|
| 1 | `test_envelope_creation` | Create `Envelope<T, S>` with data and auth signature |
| 2 | `test_envelope_verification` | Verify envelope with correct/wrong authority signatures |
| 3 | `test_verified_envelope_trust_model` | `VerifiedEnvelope` vs `TrustedEnvelope` distinctions |
| 4 | `test_envelope_bcs_roundtrip` | BCS serialization preserves envelope contents |
| 5 | `test_envelope_digest` | Envelope digest matches inner data digest |

### Wire

Add to `types/src/envelope.rs`:
```rust
#[cfg(test)]
#[path = "unit_tests/envelope_tests.rs"]
mod envelope_tests;
```

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/envelope.rs` | `crates/sui-types/src/message_envelope.rs` |
| `types/src/signature_verification.rs` | `crates/sui-types/src/signature_verification.rs` |

---

## Priority 8: Committee Tests

**Soma file to create:** `types/src/unit_tests/committee_tests.rs`
**Sui equivalent:** `crates/sui-types/src/committee.rs` inline (1 test)

### Tests to Implement

| # | Test | Description | Sui Source |
|---|------|-------------|-----------|
| 1 | `test_shuffle_by_weight` | Stake-weighted shuffle determinism and distribution | `test_shuffle_by_weight` |
| 2 | `test_committee_creation` | Committee from validators with voting power | Soma-specific |
| 3 | `test_committee_quorum_threshold` | 2f+1 quorum calculation | Soma-specific |
| 4 | `test_committee_validity_check` | Committee validity: no zero stake, total > 0 | Soma-specific |
| 5 | `test_committee_epoch_isolation` | Different epoch → different committee | Soma-specific |
| 6 | `test_voting_power_distribution` | Voting power proportional to stake | Soma-specific |

### Wire

Add to `types/src/committee.rs`:
```rust
#[cfg(test)]
#[path = "unit_tests/committee_tests.rs"]
mod committee_tests;
```

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/committee.rs` | `crates/sui-types/src/committee.rs` |

---

## Priority 9: MultiSig Tests (Extend)

**Soma file:** `types/src/unit_tests/multisig_tests.rs` (7 existing tests)
**Sui equivalent:** `crates/sui-types/src/unit_tests/multisig_tests.rs` (13 tests)

### Existing Soma Tests (7)
- `test_combine_sigs` — too many/duplicate/empty sigs
- `test_serde_roundtrip` — BCS + GenericSignature roundtrip
- `test_multisig_pk_new` — weight 0, threshold 0, dup pks validation
- `test_multisig_address` — pinned address derivation
- `test_max_sig` — max signers, unreachable threshold
- `multisig_get_pk` — get pk from MultiSig
- `multisig_get_indices` — get indices from MultiSig

### Tests to Add for Parity

| # | Test | Description | Sui Source |
|---|------|-------------|-----------|
| 1 | `test_multisig_address_pinned` | Second pinned address test with different key combination | `test_derive_multisig_address` (adapted, no zkLogin) |
| 2 | `test_multisig_verify_with_ed25519` | End-to-end sign and verify with Ed25519 keys in multisig | Pattern from Sui's multisig verification flow |
| 3 | `test_multisig_insufficient_weight` | Signature weight below threshold rejected | Pattern from Sui tests |
| 4 | `test_multisig_exact_threshold` | Exactly at threshold succeeds | Pattern from Sui tests |

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/multisig.rs` | `crates/sui-types/src/multisig.rs` |
| `types/src/unit_tests/multisig_tests.rs` | `crates/sui-types/src/unit_tests/multisig_tests.rs` |

---

## Priority 10: Genesis Builder Tests

**Soma file to create:** `types/src/unit_tests/genesis_tests.rs`
**Sui equivalent:** `crates/sui-genesis-builder/src/lib.rs` (test infrastructure, no dedicated test file)

Sui's genesis builder has tests spread across integration tests. Soma's `genesis_builder.rs` (706 lines) has zero tests.

### Tests to Implement

| # | Test | Description |
|---|------|-------------|
| 1 | `test_genesis_creates_system_state` | SystemState object exists after genesis with correct epoch (0) |
| 2 | `test_genesis_creates_validators` | All validators from config present in ValidatorSet |
| 3 | `test_genesis_creates_initial_coins` | Account allocations match TokenDistributionSchedule |
| 4 | `test_genesis_creates_seed_models` | Seed models registered in ModelRegistry |
| 5 | `test_genesis_creates_initial_targets` | Initial targets generated (if models present) |
| 6 | `test_genesis_emission_pool` | EmissionPool initialized with correct balance |
| 7 | `test_genesis_protocol_version` | Protocol version set to 1 |
| 8 | `test_genesis_deterministic` | Same config → identical genesis objects (digest match) |
| 9 | `test_genesis_builder_custom_parameters` | Custom SystemParameters respected in SystemState |
| 10 | `test_genesis_builder_multiple_validators` | 4-validator and 7-validator configurations |

### Wire

Add to `types/src/genesis_builder.rs`:
```rust
#[cfg(test)]
#[path = "unit_tests/genesis_tests.rs"]
mod genesis_tests;
```

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/genesis_builder.rs` | `crates/sui-genesis-builder/src/lib.rs` |
| `types/src/genesis.rs` | `crates/sui-types/src/genesis.rs` |
| `types/src/config/genesis_config.rs` | `crates/sui-swarm-config/src/genesis_config.rs` |

---

## Priority 11: Temporary Store Tests

**Soma file to create:** `types/src/unit_tests/temporary_store_tests.rs`
**Sui equivalent:** `crates/sui-types/src/inner_temporary_store.rs`

Soma's `temporary_store.rs` (707 lines) manages object staging during execution. Zero tests.

### Tests to Implement

| # | Test | Description |
|---|------|-------------|
| 1 | `test_temporary_store_write_and_read` | Write object to store, read it back |
| 2 | `test_temporary_store_delete` | Delete object from store |
| 3 | `test_temporary_store_mutate_existing` | Mutate object already in store |
| 4 | `test_temporary_store_written_objects` | `written_objects()` returns all created/mutated objects |
| 5 | `test_temporary_store_deleted_objects` | `deleted_objects()` returns all deleted objects |
| 6 | `test_temporary_store_input_loading` | Load owned + shared objects as inputs |
| 7 | `test_temporary_store_shared_input_versions` | Shared object version tracking via `SharedInput` |
| 8 | `test_temporary_store_revert` | Revert non-gas changes on execution failure |

### Wire

Add to `types/src/temporary_store.rs`:
```rust
#[cfg(test)]
#[path = "unit_tests/temporary_store_tests.rs"]
mod temporary_store_tests;
```

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/temporary_store.rs` | `crates/sui-types/src/inner_temporary_store.rs` |

---

## Priority 12: Storage Layer Tests

**Soma file to create:** `types/src/storage/tests.rs`
**Sui equivalent:** `crates/sui-types/src/storage/` (no dedicated tests)

### Tests to Implement

| # | Test | Description |
|---|------|-------------|
| 1 | `test_shared_in_memory_store_basic` | Insert and retrieve objects |
| 2 | `test_shared_in_memory_store_versioned_read` | Read specific version of object |
| 3 | `test_shared_in_memory_store_delete` | Delete object from store |
| 4 | `test_write_kind_variants` | `WriteKind` enum (Create, Mutate, Unwrap) coverage |
| 5 | `test_storage_error_display` | Error display formatting |

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/storage/shared_in_memory_store.rs` | `crates/sui-types/src/storage/shared_in_memory_store.rs` |
| `types/src/storage/mod.rs` | `crates/sui-types/src/storage/mod.rs` |

---

## Priority 13: Config Tests

**Soma file to create:** `types/src/config/tests.rs`
**Sui equivalent:** `crates/sui-config/` (no dedicated types-level tests)

### Tests to Implement

| # | Test | Description |
|---|------|-------------|
| 1 | `test_validator_config_builder` | `ValidatorConfigBuilder` produces valid config |
| 2 | `test_fullnode_config_builder` | `FullnodeConfigBuilder` produces valid config (no consensus) |
| 3 | `test_genesis_config_default` | Default `GenesisConfig` has sane values |
| 4 | `test_token_distribution_schedule` | Token allocation arithmetic (total matches genesis supply) |
| 5 | `test_config_persisted_roundtrip` | `PersistedConfig` write/read roundtrip |
| 6 | `test_network_config_creation` | `NetworkConfig` with multiple validators |

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/config/node_config.rs` | `crates/sui-config/src/node.rs` |
| `types/src/config/genesis_config.rs` | `crates/sui-swarm-config/src/genesis_config.rs` |
| `types/src/config/network_config.rs` | `crates/sui-config/src/` |

---

## Priority 14: Consensus Types Tests

**Soma file to create:** `types/src/consensus/tests.rs`
**Sui equivalent:** `consensus/core/src/block.rs` etc. (tests exist in consensus crate, not types)

### Tests to Implement

| # | Test | Description |
|---|------|-------------|
| 1 | `test_block_ref_ordering` | BlockRef comparison and sorting |
| 2 | `test_block_bcs_roundtrip` | Block serialization roundtrip |
| 3 | `test_commit_bcs_roundtrip` | ConsensusCommit serialization |
| 4 | `test_stake_aggregator_quorum` | StakeAggregator reaches quorum at 2f+1 |
| 5 | `test_consensus_commit_prologue` | ConsensusCommitPrologue construction |

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/consensus/block.rs` | `consensus/core/src/block.rs` |
| `types/src/consensus/commit.rs` | `consensus/core/src/commit.rs` |
| `types/src/consensus/mod.rs` | `crates/sui-types/src/messages_consensus.rs` |

---

## Priority 15: Error & Serde Tests

**Soma file to create:** `types/src/unit_tests/error_tests.rs`

### Tests to Implement

| # | Test | Description |
|---|------|-------------|
| 1 | `test_soma_error_display` | All SomaError variants have Display impl |
| 2 | `test_execution_failure_status_order` | ExecutionFailureStatus enum order stability (regression) |
| 3 | `test_soma_error_bcs_roundtrip` | Error serialization for network transport |
| 4 | `test_user_input_error_variants` | UserInputError classification (for transaction validation) |

### Cross-Reference Files

| Soma | Sui |
|------|-----|
| `types/src/error.rs` | `crates/sui-types/src/error.rs` |
| — | `crates/sui-types/src/unit_tests/execution_status_tests.rs` |

---

## Soma-Specific Coverage (Already Exists — Verify)

These test suites cover Soma-unique functionality and are **already implemented**. Verify they pass and have adequate coverage.

### system_state/unit_tests/submission_tests.rs (21 tests)
- Submission lifecycle, bond scaling, serialization, BCS roundtrip
- Target report mechanism: add, overwrite, undo, clear, quorum with varied stakes
- Transaction kind classification (SubmitData, ClaimRewards vs non-submission)
- Edge cases: zero-size data, no-model-owner, exactly-at-threshold

### system_state/unit_tests/target_tests.rs (17 tests)
- Deterministic seed/embedding generation, dimension validation
- Target generation with single/multiple models, empty model registry error
- Reward per target calculation, difficulty adjustment (high/low hit rate, bounds, bootstrap)
- Target status transitions, model selection (uniqueness, cap, seed affects)

### system_state/unit_tests/model_tests.rs (16 tests)
- Full commit-reveal lifecycle, slash on no-reveal, staking delegation
- Voluntary deactivation, min stake validation, architecture validation
- Update commit-reveal, update no-reveal cancel, update overwrite
- Commission rate (set, too-high), report quorum removal/below-quorum/undo
- Withdraw from inactive, non-owner errors, reveal epoch mismatch, non-validator report

### system_state/unit_tests/rewards_distribution_tests.rs (9 tests)
- Proportional validator rewards, large-stake subsidy, multi-epoch staking
- Tiny rewards precision, commission rates (0%, 10%, 20%)
- Reward slashing (10%, 100%), multi-staker withdrawals, 20-validator uncapped

### system_state/unit_tests/delegation_tests.rs (8 tests)
- Full add/remove stake flow, post-active withdrawal with/without rewards
- Earning rewards at last epoch, add-to-inactive error
- Preactive validator flows, staking pool exchange rate getter

### Other inline tests (30 tests)
- `model_selection.rs` (10) — KNN selection, stake weighting, voting power
- `grpc_timeout.rs` (10) — All timeout units, invalid input
- `target_state.rs` (6) — Default, counters, EMA
- `tls/mod.rs` (5) — TLS verifier variants, axum integration
- `target.rs` (4) — Deterministic embedding, seed, status
- `challenge.rs` (3) — Creation, reports, status transitions
- `multiaddr.rs` (6) — Socket addr, TCP validity, hostname
- `submission.rs` (2) — Manifest size, creation

**Action:** Run `cargo test -p types` and verify all 124 tests pass.

---

## Implementation Order

### Phase 1: Core Infrastructure Tests (Day 1-3)
1. **Base types tests** — 25 tests (Priority 1)
2. **Transaction & message tests** — 20 tests (Priority 2)
3. **Crypto & intent tests** — 9 tests (Priority 3)

### Phase 2: Effects, Object, Checkpoint (Day 4-6)
4. **Effects tests** — 15 tests (Priority 4)
5. **Object tests** — 10 tests (Priority 5)
6. **Checkpoint tests** — 7 tests (Priority 6)

### Phase 3: Auth, Committee, MultiSig (Day 7-8)
7. **Envelope & signature verification tests** — 5 tests (Priority 7)
8. **Committee tests** — 6 tests (Priority 8)
9. **MultiSig tests (extend)** — 4 new tests (Priority 9)

### Phase 4: Genesis, Storage, Config (Day 9-11)
10. **Genesis builder tests** — 10 tests (Priority 10)
11. **Temporary store tests** — 8 tests (Priority 11)
12. **Storage layer tests** — 5 tests (Priority 12)
13. **Config tests** — 6 tests (Priority 13)

### Phase 5: Consensus, Error, Finalize (Day 12-14)
14. **Consensus types tests** — 5 tests (Priority 14)
15. **Error & serde tests** — 4 tests (Priority 15)
16. **Verify all 124 existing tests still pass**
17. **Add attribution headers to all 67 derived files**

---

## Build & Run Commands

```bash
# Run all types unit tests
cargo test -p types

# Run specific test file
cargo test -p types -- base_types_tests
cargo test -p types -- transaction_tests
cargo test -p types -- crypto_tests
cargo test -p types -- effects_tests
cargo test -p types -- multisig_tests

# Run a single test
cargo test -p types -- test_bcs_owner_enum

# Run Soma-specific tests
cargo test -p types -- delegation_tests
cargo test -p types -- model_tests
cargo test -p types -- rewards_distribution
cargo test -p types -- submission_tests
cargo test -p types -- target_tests

# Check compilation only
PYO3_PYTHON=python3 cargo check -p types

# Run with output
cargo test -p types -- --nocapture 2>&1 | head -100
```

---

## Summary of New Files to Create

| File | Tests | Priority | Wired From |
|------|-------|----------|------------|
| `unit_tests/base_types_tests.rs` | 25 | P1 | `base.rs` |
| `unit_tests/transaction_tests.rs` | 20 | P2 | `transaction.rs` |
| `unit_tests/crypto_tests.rs` | 9 | P3 | `crypto.rs` |
| `unit_tests/effects_tests.rs` | 15 | P4 | `effects/mod.rs` |
| `unit_tests/object_tests.rs` | 10 | P5 | `object.rs` |
| `unit_tests/checkpoint_tests.rs` | 7 | P6 | `checkpoints.rs` |
| `unit_tests/envelope_tests.rs` | 5 | P7 | `envelope.rs` |
| `unit_tests/committee_tests.rs` | 6 | P8 | `committee.rs` |
| `unit_tests/multisig_tests.rs` (extend) | +4 | P9 | — |
| `unit_tests/genesis_tests.rs` | 10 | P10 | `genesis_builder.rs` |
| `unit_tests/temporary_store_tests.rs` | 8 | P11 | `temporary_store.rs` |
| `storage/tests.rs` | 5 | P12 | `storage/mod.rs` |
| `config/tests.rs` | 6 | P13 | `config/mod.rs` |
| `consensus/tests.rs` | 5 | P14 | `consensus/mod.rs` |
| `unit_tests/error_tests.rs` | 4 | P15 | `error.rs` |
| **Total new files** | **15** | | |
| **Total new tests** | **~139** | | |

Combined with 124 existing tests → target is **~263 tests** for the types crate.

---

## Sui Cross-Reference URLs

Key Sui files referenced in this plan:

| Category | Sui File Path |
|----------|--------------|
| Base types tests | `crates/sui-types/src/unit_tests/base_types_tests.rs` |
| Messages tests | `crates/sui-types/src/unit_tests/messages_tests.rs` |
| Crypto tests | `crates/sui-types/src/unit_tests/crypto_tests.rs` |
| Intent tests | `crates/sui-types/src/unit_tests/intent_tests.rs` |
| Effects tests | `crates/sui-types/src/unit_tests/effects_tests.rs` |
| MultiSig tests | `crates/sui-types/src/unit_tests/multisig_tests.rs` |
| Execution status tests | `crates/sui-types/src/unit_tests/execution_status_tests.rs` |
| Test utils | `crates/sui-types/src/unit_tests/utils.rs` |
| Base types source | `crates/sui-types/src/base_types.rs` |
| Transaction source | `crates/sui-types/src/transaction.rs` |
| Crypto source | `crates/sui-types/src/crypto.rs` |
| Object source | `crates/sui-types/src/object.rs` |
| Effects source | `crates/sui-types/src/effects/mod.rs` + `object_change.rs` |
| Checkpoint source | `crates/sui-types/src/messages_checkpoint.rs` |
| Envelope source | `crates/sui-types/src/message_envelope.rs` |
| Committee source | `crates/sui-types/src/committee.rs` |
| Balance change source | `crates/sui-types/src/balance_change.rs` |
| Governance source | `crates/sui-types/src/governance.rs` |
| System state | `crates/sui-types/src/sui_system_state/` |
| Genesis builder | `crates/sui-genesis-builder/src/lib.rs` |
| Genesis config | `crates/sui-swarm-config/src/genesis_config.rs` |
| Storage | `crates/sui-types/src/storage/` |
