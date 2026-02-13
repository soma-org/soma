# Protocol Config & Protocol Config Macros — Comprehensive Testing Plan

Testing plan for `protocol-config/` and `protocol-config-macros/` achieving high parity with Sui's `crates/sui-protocol-config/` and `crates/sui-protocol-config-macros/`. Covers file-by-file mapping, attribution requirements, test infrastructure, and every test needed for parity plus Soma-specific coverage.

**Sui reference**: `MystenLabs/sui` — `crates/sui-protocol-config/` and `crates/sui-protocol-config-macros/`
**Soma crates**: `protocol-config/` and `protocol-config-macros/`

---

## Audit Notes (Feb 2026)

**Priority Ranking**: #4 of 7 plans — small effort, high regression-safety value. Snapshot tests prevent accidental protocol constant changes.

**Accuracy**: Test counts are accurately reported. 8 existing tests in `tensor.rs`, 0 in `lib.rs`. The plan correctly identifies Sui's 126+ snapshot files as the primary testing mechanism and proposes an adapted version for Soma's single protocol version.

**Key Concerns**:
1. **Snapshot tests are the #1 priority** — these are cheap to implement (1 test function generates 3 snapshot files) and provide the single best regression safety mechanism for protocol config. Any accidental change to a V1 constant will be caught.
2. **`insta` dependency not yet added** — the plan correctly identifies this prerequisite. This is a dev-dependency only and should be straightforward to add.
3. **SystemParameters tests are Soma-unique and critical** — `build_system_parameters()` converts protocol config into the `SystemParameters` embedded in `SystemState`. Any conversion bug affects every epoch transition. These tests should be prioritized.
4. **Msim-specific tests are valuable** — verifying that `MAX_ALLOWED = MAX + 1` and msim overrides (gc_depth, epoch_duration) are correct prevents test infrastructure bugs from masking real issues.
5. **Low risk, high value** — this plan has the best effort-to-impact ratio of all 7 plans. Recommend implementing early.

**Estimated Effort**: ~3 engineering days as planned.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [File-by-File Cross-Reference](#file-by-file-cross-reference)
3. [Attribution Requirements](#attribution-requirements)
4. [Structural Comparison: Sui vs Soma](#structural-comparison-sui-vs-soma)
5. [Test Infrastructure](#test-infrastructure)
6. [Priority 1: Snapshot Tests (Regression Safety)](#priority-1-snapshot-tests)
7. [Priority 2: Macro-Generated Code Tests](#priority-2-macro-generated-code-tests)
8. [Priority 3: Version Management Tests](#priority-3-version-management-tests)
9. [Priority 4: Feature Flags Tests](#priority-4-feature-flags-tests)
10. [Priority 5: SystemParameters Tests](#priority-5-systemparameters-tests)
11. [Priority 6: BcsF32 & SomaTensor Tests](#priority-6-bcsf32--somatensor-tests)
12. [Priority 7: Msim-Specific Tests](#priority-7-msim-specific-tests)
13. [Priority 8: Protocol Config Macros Crate Tests](#priority-8-protocol-config-macros-crate-tests)
14. [Sui Features Intentionally Omitted](#sui-features-intentionally-omitted)
15. [Implementation Order](#implementation-order)
16. [Build & Run Commands](#build--run-commands)

---

## Executive Summary

### Current State
- **8 total tests** in the protocol-config crate (all in `tensor.rs`)
- **0 tests** in `lib.rs` — no coverage for ProtocolConfig, ProtocolVersion, Chain, FeatureFlags, SystemParameters, or macro-generated code
- **0 tests** in protocol-config-macros (proc-macro crates are tested indirectly)
- **No snapshot tests** — no regression safety for config values across versions
- **No insta dependency** — snapshot testing framework not yet added

### Target State
- **~45+ unit tests** covering all protocol config functionality
- Snapshot tests for every protocol version (currently just V1) across all chains
- Full coverage of macro-generated getters, setters, lookup, and attr_map
- ProtocolVersion arithmetic and boundary tests
- SystemParameters conversion tests
- Feature flag infrastructure tests (ready for when flags are added)
- Msim-specific behavior tests

### Test Count Summary

| Category | Sui Tests | Soma Existing | Soma Target | Gap |
|----------|-----------|---------------|-------------|-----|
| Snapshot tests (per version × chain) | 126+ snap files (3 chains × 42+ versions) | 0 | 3 (V1 × 3 chains) | 3 |
| Getter/setter tests (`test_getters`, `test_setters`) | 2 | 0 | 2 | 2 |
| Version boundary tests (`max_version_test`) | 1 | 0 | 3 | 3 |
| String lookup tests (`lookup_by_string_test`) | 1 (comprehensive) | 0 | 5 | 5 |
| Limit threshold tests (`limit_range_fn_test`) | 1 | 0 | 0 (N/A) | 0 |
| ProtocolVersion arithmetic | 0 (implicit) | 0 | 5 | 5 |
| Chain enum tests | 0 (implicit) | 0 | 3 | 3 |
| SystemParameters tests | 0 (N/A in Sui) | 0 | 5 | 5 |
| ProtocolConfig serialization | 0 (via snapshots) | 0 | 3 | 3 |
| Feature flags infrastructure | 0 (empty in V1) | 0 | 3 | 3 |
| BcsF32 tests | N/A (Soma-only) | 0 | 5 | 5 |
| SomaTensor tests | N/A (Soma-only) | 8 | 8 (existing) | 0 |
| Msim-specific tests | 0 (implicit in snapshots) | 0 | 3 | 3 |
| **Total** | **~130+** (dominated by snapshots) | **8** | **~48** | **~40** |

---

## File-by-File Cross-Reference

### Legend
- **Heavy** = Direct port/fork, needs full attribution
- **Moderate** = Significant shared patterns, needs attribution
- **Soma-only** = Original Soma code, no attribution needed

### protocol-config crate

| Soma File | Sui File | Derivation | Existing Tests | Notes |
|-----------|----------|------------|----------------|-------|
| `protocol-config/src/lib.rs` (599 lines) | `sui-protocol-config/src/lib.rs` (5263 lines) | **Heavy** | 0 | Core ProtocolConfig, ProtocolVersion, Chain, FeatureFlags. Sui has 331 Option fields + 144 feature flags; Soma has 41 Option fields + 0 feature flags. Same macro derives, same version management pattern. |
| `protocol-config/src/tensor.rs` (346 lines) | N/A | **Soma-only** | 8 | BcsF32 and SomaTensor types. No Sui equivalent. |
| `protocol-config/Cargo.toml` | `sui-protocol-config/Cargo.toml` | **Moderate** | — | Same dependency pattern but Soma uses `burn` instead of `move-vm-config`/`move-binary-format`. Soma is missing `insta` dev-dependency. |

### protocol-config-macros crate

| Soma File | Sui File | Derivation | Existing Tests | Notes |
|-----------|----------|------------|----------------|-------|
| `protocol-config-macros/src/lib.rs` (356 lines) | `sui-protocol-config-macros/src/lib.rs` (~same length) | **Heavy** (near-identical) | 0 | Three proc macros: `ProtocolConfigAccessors`, `ProtocolConfigOverride`, `ProtocolConfigFeatureFlagsGetters`. Code is functionally identical. |
| `protocol-config-macros/Cargo.toml` | `sui-protocol-config-macros/Cargo.toml` | **Heavy** | — | Same proc-macro2/syn/quote dependencies |

---

## Attribution Requirements

### Files Requiring Attribution

All files below need the following header added:

```rust
// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-protocol-config/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
```

| File | Sui Source | Derivation Level | Rationale |
|------|-----------|-----------------|-----------|
| `protocol-config/src/lib.rs` | `sui-protocol-config/src/lib.rs` | **Heavy** | ProtocolVersion struct, Chain enum, ProtocolConfig pattern (Option-wrapped fields, get_for_version, get_for_version_impl, get_for_version_if_supported, version assertions, msim fake version), FeatureFlags derive pattern, utility functions (is_false, is_empty, is_zero) — all directly adopted from Sui |
| `protocol-config-macros/src/lib.rs` | `sui-protocol-config-macros/src/lib.rs` | **Heavy** (near-identical) | All three proc macros (`ProtocolConfigAccessors`, `ProtocolConfigOverride`, `ProtocolConfigFeatureFlagsGetters`) are functionally identical, including code structure, variable names, doc comments, generated method signatures, and ProtocolConfigValue enum generation |
| `protocol-config-macros/Cargo.toml` | `sui-protocol-config-macros/Cargo.toml` | **Moderate** | Same crate structure and dependencies |

### Files NOT Requiring Attribution (Soma-only)

| File | Rationale |
|------|-----------|
| `protocol-config/src/tensor.rs` | Entirely Soma-original. BcsF32 and SomaTensor types for ML tensor serialization have no Sui equivalent. |

---

## Structural Comparison: Sui vs Soma

### Key Similarities (Adopted from Sui)

| Feature | Sui | Soma | Notes |
|---------|-----|------|-------|
| `ProtocolVersion` struct | `ProtocolVersion(u64)` with MIN/MAX/MAX_ALLOWED, new(), as_u64(), prev(), max(), Sub, Add, From<u64> | Identical | Same struct, same methods, same operators |
| `Chain` enum | `Mainnet`, `Testnet`, `Unknown` (default) with `as_str()` | Identical | Same variants, same method |
| `FeatureFlags` struct | 144 fields (bool + a few non-bool), `ProtocolConfigFeatureFlagsGetters` derive | Empty struct, same derive | Soma hasn't added feature flags yet |
| `ProtocolConfig` struct | 331 Option<T> fields, `ProtocolConfigAccessors` + `ProtocolConfigOverride` derives | 41 Option<T> fields, same derives | Same pattern but Soma has mining/model/target fields instead of Move VM fields |
| `get_for_version()` | Version bounds assertion, calls `get_for_version_impl()`, applies CONFIG_OVERRIDE + env overrides | Version bounds assertion, calls `get_for_version_impl()` | Soma omits CONFIG_OVERRIDE and env override mechanisms |
| `get_for_version_impl()` | msim fake version (MAX_ALLOWED = MAX + 1 with modified base_tx_cost_fixed), version-by-version config setup, panics on unsupported version | msim fake version (MAX_ALLOWED = MAX + 1 with modified base_fee), V1 setup, panics on unsupported version | Same pattern. Soma also has msim-specific overrides for gc_depth, epoch_duration_ms, buffer_stake |
| `get_for_version_if_supported()` | Returns Option<Self> for safe version check | Identical | Same implementation |
| `#[skip_serializing_none]` | Used on ProtocolConfig | Used on ProtocolConfig | Same serde attribute |
| `Error(pub String)` | Present | Present | Same type |
| Utility functions | `is_false()`, `is_empty()`, `is_zero()` | `is_false()`, `is_empty()`, `is_zero()` | Identical implementations |

### Key Differences (Soma-specific)

| Feature | Sui | Soma | Impact on Testing |
|---------|-----|------|-------------------|
| `SystemParameters` struct | N/A (Sui uses Move objects) | 25+ concrete fields with `build_system_parameters()` method | Needs dedicated tests |
| `BcsF32` type | N/A | Float wrapper for BCS serialization | Needs tests (in tensor.rs) |
| `SomaTensor` type | N/A | TensorData wrapper with BCS serialization | Has 8 tests (adequate) |
| `CONFIG_OVERRIDE` mechanism | thread_local OverrideFn + env var override + `apply_overrides_for_testing()` + `OverrideGuard` | **Not present** | May want to add for test flexibility |
| `poison_get_for_min_version` | AtomicBool guard preventing `get_for_min_version()` on validators | **Not present** | May want to add for safety |
| `get_for_min_version()` | Convenience for client code (with poison guard) | **Not present** | Consider adding |
| `get_for_max_version_UNSAFE()` | Convenience for genesis (with poison guard) | **Not present** | Consider adding |
| `LimitThresholdCrossed` enum | Soft/Hard/None limit checking | **Not present** | N/A (Move VM specific) |
| `check_limit!` / `check_limit_by_meter!` macros | Limit range checking for execution | **Not present** | N/A (Move VM specific) |
| `VerifierConfig` / `BinaryConfig` | Move bytecode verifier configuration | **Not present** | N/A (no Move VM in Soma) |
| Feature flag test setters | 39 manual `set_*_for_testing()` methods | None (empty FeatureFlags) | Not needed until flags are added |
| Protocol versions | V1 through V112 | V1 only | Far fewer snapshots needed |

---

## Test Infrastructure

### Dependencies to Add

**`protocol-config/Cargo.toml`** — Add dev-dependencies:
```toml
[dev-dependencies]
insta = { version = "1.41", features = ["yaml"] }
serde_yaml = "0.9"
bcs = { workspace = true }  # Already a dependency, but verify it's available in tests
```

### Snapshot Directory

Create: `protocol-config/src/snapshots/` (insta default location)

Snapshots will be auto-generated on first run with `cargo insta review` or `INSTA_UPDATE=always`.

### Test Module Structure

Add to `protocol-config/src/lib.rs` at the bottom:

```rust
#[cfg(all(test, not(msim)))]
mod test {
    use super::*;
    // ... tests here
}
```

Note: Sui uses `#[cfg(all(test, not(msim)))]` to prevent snapshot tests from running under the simulator (where MAX_ALLOWED differs). Soma should follow this pattern.

---

## Priority 1: Snapshot Tests

**Sui equivalent:** `snapshot_tests()` in `sui-protocol-config/src/lib.rs`
**Sui snapshot count:** 126+ files across `crates/sui-protocol-config/src/snapshots/`

### Why Snapshots Matter

Snapshot tests are the **primary regression safety mechanism** for protocol config. They ensure:
1. No accidental changes to existing protocol version constants
2. Chain-specific overrides are captured and reviewed
3. New versions are intentionally added (never silently modified)

Sui's test has this prominent warning:
```
IMPORTANT: never update snapshots from this test. only add new versions!
```

### Tests to Implement

| Test | Description | Sui Equivalent |
|------|-------------|---------------|
| `snapshot_tests` | For each chain (Unknown, Mainnet, Testnet) × each version (1..=MAX), create a YAML snapshot via `insta::assert_yaml_snapshot!`. This generates one `.snap` file per (chain, version) pair. | `snapshot_tests()` — identical pattern |

**Implementation:**

```rust
#[test]
fn snapshot_tests() {
    println!("\n============================================================================");
    println!("!                                                                          !");
    println!("! IMPORTANT: never update snapshots from this test. only add new versions! !");
    println!("!                                                                          !");
    println!("============================================================================\n");
    for chain_id in &[Chain::Unknown, Chain::Mainnet, Chain::Testnet] {
        let chain_str = match chain_id {
            Chain::Unknown => "".to_string(),
            _ => format!("{:?}_", chain_id),
        };
        for i in MIN_PROTOCOL_VERSION..=MAX_PROTOCOL_VERSION {
            let cur = ProtocolVersion::new(i);
            insta::assert_yaml_snapshot!(
                format!("{}version_{}", chain_str, cur.as_u64()),
                ProtocolConfig::get_for_version(cur, *chain_id)
            );
        }
    }
}
```

**Expected snapshot files (3 for V1):**
- `protocol_config__test__version_1.snap`
- `protocol_config__test__Mainnet_version_1.snap`
- `protocol_config__test__Testnet_version_1.snap`

**Notes:**
- `ProtocolConfig` already derives `Serialize`, so YAML snapshot serialization will work
- The `SomaTensor` and `BcsF32` types need `Serialize` support (already present)
- Snapshot content will include all 41 Option fields + version + feature_flags
- When new protocol versions are added, new snapshot files are added — **existing ones must never be updated**

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `protocol-config/src/lib.rs` (add test module) | `sui-protocol-config/src/lib.rs` (lines 5068-5097) |
| `protocol-config/src/snapshots/` (create) | `sui-protocol-config/src/snapshots/` (126+ files) |

---

## Priority 2: Macro-Generated Code Tests

**Sui equivalent:** `test_getters()`, `test_setters()`, `lookup_by_string_test()` in `sui-protocol-config/src/lib.rs`

These tests verify that the proc macros in `protocol-config-macros` generate correct code when applied to `ProtocolConfig` and `FeatureFlags`.

### Tests to Implement

| Test | Description | Sui Equivalent |
|------|-------------|---------------|
| `test_getters` | Verify that auto-generated getter and `_as_option` getter return consistent values for a known field | `test_getters()` |
| `test_setters` | Verify `set_*_for_testing()`, `set_*_from_str_for_testing()`, `disable_*_for_testing()`, and `set_attr_for_testing()` all work correctly | `test_setters()` |
| `test_lookup_attr_existing` | Look up a known field by string name, verify it returns `Some(ProtocolConfigValue::u64(expected))` | Part of `lookup_by_string_test()` |
| `test_lookup_attr_nonexistent` | Look up a random/unknown string, verify it returns `None` | Part of `lookup_by_string_test()` |
| `test_attr_map_completeness` | Verify `attr_map()` returns all 41 fields, all non-None for V1 | Part of `lookup_by_string_test()` |

**Implementation details:**

```rust
#[test]
fn test_getters() {
    let prot = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    // Verify getter and as_option getter are consistent
    assert_eq!(prot.base_fee(), prot.base_fee_as_option().unwrap());
    assert_eq!(prot.epoch_duration_ms(), prot.epoch_duration_ms_as_option().unwrap());
    assert_eq!(prot.target_embedding_dim(), prot.target_embedding_dim_as_option().unwrap());
}

#[test]
fn test_setters() {
    let mut prot = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);

    // Test direct setter
    prot.set_base_fee_for_testing(9999);
    assert_eq!(prot.base_fee(), 9999);

    // Test string setter
    prot.set_base_fee_from_str_for_testing("1234".to_string());
    assert_eq!(prot.base_fee(), 1234);

    // Test disable (set to None)
    prot.disable_base_fee_for_testing();
    assert_eq!(prot.base_fee_as_option(), None);

    // Test set_attr_for_testing (generic string-based setter)
    prot.set_attr_for_testing("base_fee".to_string(), "5678".to_string());
    assert_eq!(prot.base_fee(), 5678);
}

#[test]
fn test_lookup_attr_existing() {
    let prot = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    assert_eq!(
        prot.lookup_attr("base_fee".to_string()),
        Some(ProtocolConfigValue::u64(prot.base_fee()))
    );
    // Test a u32 field
    assert_eq!(
        prot.lookup_attr("consensus_gc_depth".to_string()),
        Some(ProtocolConfigValue::u32(prot.consensus_gc_depth()))
    );
}

#[test]
fn test_lookup_attr_nonexistent() {
    let prot = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    assert!(prot.lookup_attr("nonexistent_field".to_string()).is_none());
    assert!(prot.lookup_attr("".to_string()).is_none());
}

#[test]
fn test_attr_map_completeness() {
    let prot = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    let map = prot.attr_map();
    // All 41 Option<T> fields should be in the map
    // (version and feature_flags are not Option<T>, so they're excluded)
    assert!(map.contains_key("base_fee"));
    assert!(map.contains_key("epoch_duration_ms"));
    assert!(map.contains_key("target_embedding_dim"));
    assert!(map.contains_key("max_submission_data_size"));
    // In V1, all fields should be Some (none are None)
    for (key, value) in &map {
        assert!(value.is_some(), "Field '{}' is None in V1 but should be Some", key);
    }
}
```

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `protocol-config/src/lib.rs` (add tests) | `sui-protocol-config/src/lib.rs` (lines 5099-5221) |
| `protocol-config-macros/src/lib.rs` (no changes, tested indirectly) | `sui-protocol-config-macros/src/lib.rs` |

---

## Priority 3: Version Management Tests

**Sui equivalent:** `max_version_test()` in `sui-protocol-config/src/lib.rs`

### Tests to Implement

| Test | Description | Sui Equivalent |
|------|-------------|---------------|
| `test_get_for_version_v1` | `get_for_version(1, Unknown)` succeeds and returns correct version field | N/A (implicit in other tests) |
| `test_max_version_panics` | `get_for_version_impl(MAX_PROTOCOL_VERSION + 1, Unknown)` panics with "unsupported version" | `max_version_test()` |
| `test_get_for_version_if_supported_valid` | `get_for_version_if_supported(1, Unknown)` returns `Some` | N/A |
| `test_get_for_version_if_supported_invalid` | `get_for_version_if_supported(0, Unknown)` and `(MAX+1, Unknown)` return `None` | N/A |
| `test_get_for_version_below_min_panics` | `get_for_version(0, Unknown)` panics on min version assertion | N/A (implicit in Sui's assertions) |

**Implementation details:**

```rust
#[test]
fn test_get_for_version_v1() {
    let config = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    assert_eq!(config.version.as_u64(), 1);
    // Verify a few known V1 values
    assert_eq!(config.base_fee(), 1000);
    assert_eq!(config.target_embedding_dim(), 768);
    assert_eq!(config.epoch_duration_ms(), 24 * 60 * 60);
}

#[test]
#[should_panic(expected = "unsupported version")]
fn test_max_version_panics() {
    // When this does not panic, version higher than MAX_PROTOCOL_VERSION exists.
    // To fix, bump MAX_PROTOCOL_VERSION or add support for the new version.
    let _ = ProtocolConfig::get_for_version_impl(
        ProtocolVersion::new(MAX_PROTOCOL_VERSION + 1),
        Chain::Unknown,
    );
}

#[test]
fn test_get_for_version_if_supported_valid() {
    let result = ProtocolConfig::get_for_version_if_supported(ProtocolVersion::new(1), Chain::Unknown);
    assert!(result.is_some());
    assert_eq!(result.unwrap().version.as_u64(), 1);
}

#[test]
fn test_get_for_version_if_supported_too_high() {
    // Version beyond MAX_ALLOWED should return None (not panic)
    let result = ProtocolConfig::get_for_version_if_supported(
        ProtocolVersion::new(MAX_PROTOCOL_VERSION + 100),
        Chain::Unknown,
    );
    assert!(result.is_none());
}

#[test]
#[should_panic]
fn test_get_for_version_below_min_panics() {
    let _ = ProtocolConfig::get_for_version(ProtocolVersion::new(0), Chain::Unknown);
}
```

**Note:** The `test_max_version_panics` test needs access to `get_for_version_impl` which is currently private. Either:
1. Make it `pub(crate)` (to allow test module access), or
2. Test through `get_for_version` (which also panics, but with a different message about MAX_ALLOWED vs "unsupported version")

Sui's test calls `get_for_version_impl` directly. Since Soma's test module is inside `lib.rs` (same module), it should have access to private functions.

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `protocol-config/src/lib.rs` | `sui-protocol-config/src/lib.rs` (lines 5126-5135) |

---

## Priority 4: Feature Flags Tests

**Sui equivalent:** Feature flag portion of `lookup_by_string_test()` in `sui-protocol-config/src/lib.rs`

Soma's `FeatureFlags` is currently empty (`struct FeatureFlags {}`), but the macro infrastructure is in place. These tests verify the infrastructure works correctly and will serve as a foundation when feature flags are added.

### Tests to Implement

| Test | Description | Sui Equivalent |
|------|-------------|---------------|
| `test_feature_flags_lookup_nonexistent` | `lookup_feature("random string")` returns `None` | Part of `lookup_by_string_test()` |
| `test_feature_flags_map_empty` | `feature_map()` returns empty BTreeMap (since no flags exist yet) | Part of `lookup_by_string_test()` |
| `test_feature_flags_default` | Default FeatureFlags is empty / all false | N/A |

**Implementation details:**

```rust
#[test]
fn test_feature_flags_lookup_nonexistent() {
    let prot = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    assert!(prot.lookup_feature("some random string".to_string()).is_none());
    assert!(prot.lookup_feature("package_upgrades".to_string()).is_none()); // Sui has this, Soma doesn't
}

#[test]
fn test_feature_flags_map_empty() {
    let prot = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    let map = prot.feature_map();
    // Currently no feature flags defined in Soma
    assert!(map.is_empty());
}

#[test]
fn test_feature_flags_default() {
    let flags = FeatureFlags::default();
    // Verify default is empty (no fields to check yet)
    let map = flags.attr_map();
    assert!(map.is_empty());
}
```

**When feature flags are added:** Update these tests to verify:
- Default value (false) in early versions
- Enabled (true) in later versions
- String lookup returns correct value
- Feature map contains the new flag

### Cross-Reference Files
| Soma | Sui |
|------|-----|
| `protocol-config/src/lib.rs` (FeatureFlags struct) | `sui-protocol-config/src/lib.rs` (lines 383-991, 144 flags) |

---

## Priority 5: SystemParameters Tests

**Soma-only** — No Sui equivalent (Sui uses Move system objects instead).

### Tests to Implement

| Test | Description |
|------|-------------|
| `test_build_system_parameters_v1` | `build_system_parameters(None)` returns correct values matching V1 config |
| `test_build_system_parameters_with_current_fee` | `build_system_parameters(Some(50))` uses the provided fee instead of initial |
| `test_build_system_parameters_without_current_fee` | `build_system_parameters(None)` falls back to `initial_value_fee_bps` |
| `test_system_parameters_serialization` | BCS round-trip of SystemParameters preserves all fields |
| `test_system_parameters_tensor_fields` | Distance threshold and embedding fields are correct SomaTensor scalars |

**Implementation details:**

```rust
#[test]
fn test_build_system_parameters_v1() {
    let config = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    let params = config.build_system_parameters(None);

    assert_eq!(params.epoch_duration_ms, 24 * 60 * 60);
    assert_eq!(params.validator_reward_allocation_bps, 7000);
    assert_eq!(params.base_fee, 1000);
    assert_eq!(params.write_object_fee, 300);
    assert_eq!(params.value_fee_bps, 10); // Falls back to initial_value_fee_bps
    assert_eq!(params.target_models_per_target, 3);
    assert_eq!(params.target_embedding_dim, 768);
    assert_eq!(params.target_initial_targets_per_epoch, 20);
    assert_eq!(params.submission_bond_per_byte, 10);
    assert_eq!(params.challenger_bond_per_byte, 5);
    assert_eq!(params.max_submission_data_size, 1024 * 1024 * 1024);
}

#[test]
fn test_build_system_parameters_with_current_fee() {
    let config = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    let params = config.build_system_parameters(Some(50));
    // Should use the provided current fee, not initial_value_fee_bps
    assert_eq!(params.value_fee_bps, 50);
}

#[test]
fn test_build_system_parameters_without_current_fee() {
    let config = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    let params = config.build_system_parameters(None);
    // Should fall back to initial_value_fee_bps (10)
    assert_eq!(params.value_fee_bps, 10);
}

#[test]
fn test_system_parameters_serialization() {
    let config = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    let params = config.build_system_parameters(None);
    let serialized = bcs::to_bytes(&params).unwrap();
    let deserialized: SystemParameters = bcs::from_bytes(&serialized).unwrap();
    assert_eq!(params, deserialized);
}

#[test]
fn test_system_parameters_tensor_fields() {
    let config = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    let params = config.build_system_parameters(None);

    // Distance thresholds should be scalar tensors
    assert_eq!(params.target_initial_distance_threshold.shape(), &[1]);
    assert_eq!(params.target_initial_distance_threshold.as_scalar(), 0.5);

    assert_eq!(params.target_max_distance_threshold.shape(), &[1]);
    assert_eq!(params.target_max_distance_threshold.as_scalar(), 1.0);

    assert_eq!(params.target_min_distance_threshold.shape(), &[1]);
    assert_eq!(params.target_min_distance_threshold.as_scalar(), 0.1);
}
```

---

## Priority 6: BcsF32 & SomaTensor Tests

### BcsF32 Tests — NEW (in tensor.rs)

The existing 8 tests cover SomaTensor well but BcsF32 has **zero dedicated tests**.

| Test | Description |
|------|-------------|
| `test_bcsf32_creation` | `BcsF32::new(3.14)` and `value()` round-trip |
| `test_bcsf32_serialization` | BCS round-trip preserves value |
| `test_bcsf32_from_str` | `FromStr` parsing works ("3.14" -> BcsF32(3.14)) |
| `test_bcsf32_display` | Display format matches f32 display |
| `test_bcsf32_special_values` | 0.0, -0.0, INFINITY, NEG_INFINITY, NaN serialization |

**Implementation details:**

```rust
#[test]
fn test_bcsf32_creation() {
    let f = BcsF32::new(3.14);
    assert_eq!(f.value(), 3.14);

    let f2: BcsF32 = 2.71.into();
    assert_eq!(f2.value(), 2.71);

    let f3: f32 = f2.into();
    assert_eq!(f3, 2.71);
}

#[test]
fn test_bcsf32_serialization() {
    let original = BcsF32::new(std::f32::consts::PI);
    let bytes = bcs::to_bytes(&original).unwrap();
    let restored: BcsF32 = bcs::from_bytes(&bytes).unwrap();
    assert_eq!(original.value(), restored.value());
}

#[test]
fn test_bcsf32_from_str() {
    let f: BcsF32 = "3.14".parse().unwrap();
    assert!((f.value() - 3.14).abs() < 1e-6);

    let bad: Result<BcsF32, _> = "not_a_number".parse();
    assert!(bad.is_err());
}

#[test]
fn test_bcsf32_display() {
    let f = BcsF32::new(42.5);
    assert_eq!(format!("{}", f), "42.5");
}

#[test]
fn test_bcsf32_special_values() {
    for &val in &[0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY] {
        let f = BcsF32::new(val);
        let bytes = bcs::to_bytes(&f).unwrap();
        let restored: BcsF32 = bcs::from_bytes(&bytes).unwrap();
        assert_eq!(f.value().to_bits(), restored.value().to_bits());
    }
}
```

### Existing SomaTensor Tests — ADEQUATE

The 8 existing tests in `tensor.rs` cover:
- Creation, scalar, zeros
- Hash equality/inequality
- BCS round-trip serialization (regular, scalar, special values)
- TensorData conversion

**No additional SomaTensor tests needed** unless new methods are added.

---

## Priority 7: Msim-Specific Tests

**Sui equivalent:** Implicit in snapshot tests (excluded via `#[cfg(all(test, not(msim)))]`) and tested via e2e protocol version upgrade tests.

These tests verify msim-specific behavior that is critical for the test infrastructure.

### Tests to Implement

These should be in a **separate test module** guarded by `#[cfg(all(test, msim))]`:

| Test | Description |
|------|-------------|
| `test_msim_max_allowed_is_max_plus_one` | Under msim, `ProtocolVersion::MAX_ALLOWED.as_u64() == MAX_PROTOCOL_VERSION + 1` |
| `test_msim_fake_version_has_different_base_fee` | The msim fake version has `base_fee = V1_base_fee + 1000` |
| `test_msim_overrides_applied` | Under msim, `consensus_gc_depth == 5`, `epoch_duration_ms == 60000`, `buffer_stake == 0` |

**Implementation details:**

```rust
#[cfg(all(test, msim))]
mod msim_tests {
    use super::*;

    #[test]
    fn test_msim_max_allowed_is_max_plus_one() {
        assert_eq!(ProtocolVersion::MAX_ALLOWED.as_u64(), MAX_PROTOCOL_VERSION + 1);
    }

    #[test]
    fn test_msim_fake_version_has_different_base_fee() {
        let v1 = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
        let fake = ProtocolConfig::get_for_version(ProtocolVersion::MAX_ALLOWED, Chain::Unknown);
        assert_eq!(fake.base_fee(), v1.base_fee() + 1000);
    }

    #[test]
    fn test_msim_overrides_applied() {
        let config = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
        assert_eq!(config.consensus_gc_depth(), 5);
        assert_eq!(config.epoch_duration_ms(), 1000 * 60);
        assert_eq!(config.buffer_stake_for_protocol_upgrade_bps(), 0);
    }
}
```

**Build requirement:** These tests must be compiled and run with `RUSTFLAGS="--cfg msim"`:
```bash
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p protocol-config
```

---

## Priority 8: Protocol Config Macros Crate Tests

**Sui equivalent:** No direct tests in `sui-protocol-config-macros/` (proc-macro crates are tested indirectly through the consuming crate).

### Approach

Proc-macro crates cannot contain `#[test]` functions that test the macros directly (the macro runs at compile time). Testing is done indirectly through the `protocol-config` crate tests in Priorities 2-4 above.

However, we can add **compile-time validation tests** using `trybuild` or similar:

| Test | Description | Priority |
|------|-------------|----------|
| Verify `ProtocolConfigAccessors` generates correct getters | Covered by Priority 2 tests | Already planned |
| Verify `ProtocolConfigOverride` generates `ProtocolConfigOptional` | Test that `ProtocolConfigOptional` struct exists and `apply_to()` works | Low |
| Verify `ProtocolConfigFeatureFlagsGetters` on empty struct | Covered by Priority 4 tests | Already planned |

### Optional: ProtocolConfigOptional Test

```rust
#[test]
fn test_protocol_config_optional_exists() {
    // Verify the ProtocolConfigOverride macro generated the Optional struct
    // This is a compile-time check — if this compiles, the macro worked
    let _: ProtocolConfigOptional;
}
```

**Note:** `ProtocolConfigOptional` is not currently used in Soma's `get_for_version()` (unlike Sui which uses it for env var overrides). If/when `CONFIG_OVERRIDE` or env var override support is added, this test becomes important.

---

## ProtocolVersion Arithmetic Tests — NEW

These test the `ProtocolVersion` struct's operators and methods.

| Test | Description |
|------|-------------|
| `test_protocol_version_new_and_as_u64` | Roundtrip: `ProtocolVersion::new(42).as_u64() == 42` |
| `test_protocol_version_from_u64` | `ProtocolVersion::from(5)` equals `ProtocolVersion::new(5)` |
| `test_protocol_version_add` | `ProtocolVersion::new(3) + 2 == ProtocolVersion::new(5)` |
| `test_protocol_version_sub` | `ProtocolVersion::new(5) - 2 == ProtocolVersion::new(3)` |
| `test_protocol_version_prev` | `ProtocolVersion::new(3).prev() == ProtocolVersion::new(2)` |

```rust
#[test]
fn test_protocol_version_new_and_as_u64() {
    let v = ProtocolVersion::new(42);
    assert_eq!(v.as_u64(), 42);
}

#[test]
fn test_protocol_version_from_u64() {
    let v: ProtocolVersion = 5u64.into();
    assert_eq!(v, ProtocolVersion::new(5));
}

#[test]
fn test_protocol_version_add() {
    let v = ProtocolVersion::new(3) + 2;
    assert_eq!(v, ProtocolVersion::new(5));
}

#[test]
fn test_protocol_version_sub() {
    let v = ProtocolVersion::new(5) - 2;
    assert_eq!(v, ProtocolVersion::new(3));
}

#[test]
fn test_protocol_version_prev() {
    let v = ProtocolVersion::new(3).prev();
    assert_eq!(v, ProtocolVersion::new(2));
}
```

---

## Chain Enum Tests — NEW

| Test | Description |
|------|-------------|
| `test_chain_as_str` | Verify `as_str()` for all variants |
| `test_chain_default` | `Chain::default() == Chain::Unknown` |
| `test_chain_serialization` | BCS/serde round-trip for all variants |

```rust
#[test]
fn test_chain_as_str() {
    assert_eq!(Chain::Mainnet.as_str(), "mainnet");
    assert_eq!(Chain::Testnet.as_str(), "testnet");
    assert_eq!(Chain::Unknown.as_str(), "unknown");
}

#[test]
fn test_chain_default() {
    assert_eq!(Chain::default(), Chain::Unknown);
}

#[test]
fn test_chain_serialization() {
    for chain in [Chain::Mainnet, Chain::Testnet, Chain::Unknown] {
        let serialized = bcs::to_bytes(&chain).unwrap();
        let deserialized: Chain = bcs::from_bytes(&serialized).unwrap();
        assert_eq!(chain, deserialized);
    }
}
```

---

## ProtocolConfig BCS Serialization Tests — NEW

| Test | Description |
|------|-------------|
| `test_protocol_config_bcs_roundtrip` | Full ProtocolConfig BCS serialization round-trip |
| `test_protocol_config_deterministic` | Same version + chain always produces identical serialized bytes |
| `test_protocol_config_with_none_fields` | Config with Some fields set to None serializes correctly |

```rust
#[test]
fn test_protocol_config_bcs_roundtrip() {
    let config = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    let bytes = bcs::to_bytes(&config).unwrap();
    // Note: ProtocolConfig uses #[skip_serializing_none] which is serde-specific.
    // BCS may not respect this. If BCS round-trip fails, test with serde_json instead.
    assert!(!bytes.is_empty());
}

#[test]
fn test_protocol_config_deterministic() {
    let config1 = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    let config2 = ProtocolConfig::get_for_version(ProtocolVersion::new(1), Chain::Unknown);
    let bytes1 = bcs::to_bytes(&config1).unwrap();
    let bytes2 = bcs::to_bytes(&config2).unwrap();
    assert_eq!(bytes1, bytes2);
}
```

---

## Sui Features Intentionally Omitted

The following Sui features are **not present in Soma** and do not need test coverage:

| Sui Feature | Reason Not Needed |
|-------------|-------------------|
| `CONFIG_OVERRIDE` / `apply_overrides_for_testing()` / `OverrideGuard` | Soma doesn't have runtime config override mechanism. Consider adding if test flexibility becomes an issue. |
| `poison_get_for_min_version()` | Soma doesn't have `get_for_min_version()` or `get_for_max_version_UNSAFE()`. Consider adding for client safety. |
| `LimitThresholdCrossed` enum / `check_limit!` / `check_limit_by_meter!` macros | Move VM execution limit checking. Not applicable to Soma. |
| `VerifierConfig` / `BinaryConfig` / Move binary format config | Move bytecode verifier. Not applicable to Soma. |
| Move-related ProtocolConfig fields (max_arguments, max_modules_in_publish, max_move_object_size, etc.) | 290+ fields related to Move VM, gas cost tables, storage costs, object runtime limits. Not applicable. |
| Feature flag test setters (39 manual methods) | Soma has no feature flags yet. Will be needed when flags are added. |
| `ConsensusTransactionOrdering` / `PerObjectCongestionControlMode` / `ConsensusChoice` / `ConsensusNetwork` enums | Sui-specific enum types used in feature flags. Not applicable. |
| zklogin / passkey / bridge feature flags | Sui-specific authentication and bridge features. Not applicable. |

---

## Implementation Order

### Phase 1: Infrastructure Setup (Day 1)
1. Add `insta` (with "yaml" feature) to `protocol-config/Cargo.toml` dev-dependencies
2. Create `protocol-config/src/snapshots/` directory
3. Add `#[cfg(all(test, not(msim)))] mod test { ... }` block to `lib.rs`
4. Add attribution headers to `protocol-config/src/lib.rs` and `protocol-config-macros/src/lib.rs`

### Phase 2: Snapshot Tests (Day 1)
5. Implement `snapshot_tests` — generates 3 snapshot files
6. Run `cargo insta review` to accept initial snapshots
7. Verify snapshots contain all 41 fields with correct V1 values

### Phase 3: Core Tests (Day 2)
8. ProtocolVersion arithmetic tests (5 tests)
9. Chain enum tests (3 tests)
10. Version management tests (5 tests)
11. Macro-generated code tests: getters, setters, lookup (5 tests)

### Phase 4: Feature & System Tests (Day 2)
12. Feature flags infrastructure tests (3 tests)
13. SystemParameters tests (5 tests)
14. BcsF32 tests (5 tests)
15. ProtocolConfig serialization tests (2-3 tests)

### Phase 5: Msim Tests (Day 3)
16. Add `#[cfg(all(test, msim))] mod msim_tests { ... }` block
17. Implement 3 msim-specific tests
18. Run with `RUSTFLAGS="--cfg msim"` to verify

### Phase 6: Verification (Day 3)
19. Run all non-msim tests: `cargo test -p protocol-config`
20. Run msim tests: `PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p protocol-config`
21. Verify snapshot files are committed to git
22. Verify attribution headers are present

---

## Build & Run Commands

```bash
# Run all protocol-config unit tests (non-msim)
cargo test -p protocol-config

# Run specific test
cargo test -p protocol-config -- test_getters
cargo test -p protocol-config -- snapshot_tests

# Run with insta update (first time or when adding new versions)
INSTA_UPDATE=always cargo test -p protocol-config -- snapshot_tests

# Review snapshots interactively
cargo insta review -p protocol-config

# Run msim-specific tests
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p protocol-config

# Check compilation of protocol-config-macros
cargo check -p protocol-config-macros

# Run all tests with output
cargo test -p protocol-config -- --nocapture
```

---

## Summary of Changes Needed

### New Files to Create

| File | Purpose |
|------|---------|
| `protocol-config/src/snapshots/protocol_config__test__version_1.snap` | V1 snapshot (Unknown chain) |
| `protocol-config/src/snapshots/protocol_config__test__Mainnet_version_1.snap` | V1 snapshot (Mainnet) |
| `protocol-config/src/snapshots/protocol_config__test__Testnet_version_1.snap` | V1 snapshot (Testnet) |

### Files to Modify

| File | Changes |
|------|---------|
| `protocol-config/Cargo.toml` | Add `[dev-dependencies]` with `insta` |
| `protocol-config/src/lib.rs` | Add attribution header + `#[cfg(test)]` module with ~35 tests + `#[cfg(msim)]` module with 3 tests |
| `protocol-config/src/tensor.rs` | Add 5 BcsF32 tests to existing test module |
| `protocol-config-macros/src/lib.rs` | Add attribution header only (no code changes) |
| `protocol-config-macros/Cargo.toml` | Add attribution header comment only |

### Test Count Breakdown

| Module | New Tests | Existing Tests | Total |
|--------|-----------|----------------|-------|
| `lib.rs` — snapshot | 1 (generates 3 snap files) | 0 | 1 |
| `lib.rs` — macro tests | 5 | 0 | 5 |
| `lib.rs` — version tests | 5 | 0 | 5 |
| `lib.rs` — feature flag tests | 3 | 0 | 3 |
| `lib.rs` — SystemParameters tests | 5 | 0 | 5 |
| `lib.rs` — ProtocolVersion arithmetic | 5 | 0 | 5 |
| `lib.rs` — Chain tests | 3 | 0 | 3 |
| `lib.rs` — serialization tests | 2 | 0 | 2 |
| `lib.rs` — msim tests | 3 | 0 | 3 |
| `tensor.rs` — BcsF32 tests | 5 | 0 | 5 |
| `tensor.rs` — SomaTensor tests | 0 | 8 | 8 |
| **Total** | **37** | **8** | **45** |

---

## Sui Cross-Reference Summary

| Category | Sui File | Soma File | Key Differences |
|----------|----------|-----------|-----------------|
| Main config | `crates/sui-protocol-config/src/lib.rs` (5263 lines, 331 Option fields, 144 feature flags) | `protocol-config/src/lib.rs` (599 lines, 41 Option fields, 0 feature flags) | Soma is much smaller (no Move VM fields). Same macro/version pattern. |
| Proc macros | `crates/sui-protocol-config-macros/src/lib.rs` | `protocol-config-macros/src/lib.rs` | **Nearly identical** — same 3 derive macros with same code structure |
| Snapshots | `crates/sui-protocol-config/src/snapshots/` (126+ files, V1-V112 × 3 chains) | None (to be created) | Soma will have 3 snap files (V1 × 3 chains) |
| Sui tests | 6 tests: `snapshot_tests`, `test_getters`, `test_setters`, `max_version_test`, `lookup_by_string_test`, `limit_range_fn_test` | 8 tests in tensor.rs only | Soma will add 37 new tests for full coverage |
| Tensor types | N/A | `protocol-config/src/tensor.rs` (BcsF32, SomaTensor) | Soma-only — ML tensor serialization |
| SystemParameters | N/A (Sui uses Move system objects) | `SystemParameters` struct with `build_system_parameters()` | Soma-only convenience struct |
| Config override | `CONFIG_OVERRIDE` thread_local + env var + `OverrideGuard` | Not present | Consider adding for test flexibility |
| Limit checking | `LimitThresholdCrossed` + `check_limit!` macros | Not present | Move VM specific, not needed |
