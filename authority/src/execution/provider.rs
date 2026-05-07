// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Executor for the on-chain provider registry.
//!
//! Two ops:
//!
//! | Op                | Caller                    | Purpose                                       |
//! |-------------------|---------------------------|-----------------------------------------------|
//! | RegisterProvider  | anyone (becomes provider) | Creates a Provider object at derive_id(signer)|
//! | UpdateProvider    | provider.address          | Bumps registered_at_ms; may change endpoint   |
//!
//! `UpdateProvider` doubles as the heartbeat — there is no separate
//! tx kind. Off-chain code decides whether a provider is "live" by
//! looking at `registered_at_ms` against a staleness threshold.

use types::base::{SomaAddress, TimestampMs};
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::ExecutionResult;
use types::object::{Object, ObjectID};
use types::provider::Provider;
use types::temporary_store::TemporaryStore;
use types::transaction::{RegisterProviderArgs, TransactionKind, UpdateProviderArgs};

use super::TransactionExecutor;

/// Maximum allowed length for `Provider.endpoint`. Bounded to keep
/// the on-chain object small; the URL is just a routing pointer
/// anyway. 1024 is generous (typical OpenAI-style endpoints are
/// well under 100 chars).
const PROVIDER_ENDPOINT_MAX_LEN: usize = 1024;

pub struct ProviderExecutor;

impl ProviderExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }

    fn read_clock_ts(store: &TemporaryStore) -> ExecutionResult<TimestampMs> {
        store
            .read_clock_timestamp_ms()
            .ok_or(ExecutionFailureStatus::ProviderClockMissing)
    }

    /// Validate a candidate `endpoint` string. Empty or oversized
    /// endpoints are rejected at execution.
    fn validate_endpoint(endpoint: &str) -> ExecutionResult<()> {
        if endpoint.is_empty() {
            return Err(ExecutionFailureStatus::ProviderInvalidEndpoint {
                reason: "endpoint must be non-empty".to_string(),
            }
            .into());
        }
        if endpoint.len() > PROVIDER_ENDPOINT_MAX_LEN {
            return Err(ExecutionFailureStatus::ProviderInvalidEndpoint {
                reason: format!(
                    "endpoint exceeds max length {} (got {})",
                    PROVIDER_ENDPOINT_MAX_LEN,
                    endpoint.len(),
                ),
            }
            .into());
        }
        Ok(())
    }

    /// Execute `RegisterProvider`. Hard-fails on duplicate.
    fn execute_register(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: RegisterProviderArgs,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        Self::validate_endpoint(&args.endpoint)?;

        // Re-registration check. The Provider object's id is
        // deterministic from the signer's address, so existence is a
        // direct lookup — no scan.
        let provider_id = Provider::derive_id(signer);
        if store.read_object(&provider_id).is_some() {
            return Err(ExecutionFailureStatus::ProviderAlreadyExists.into());
        }

        let now_ms = Self::read_clock_ts(store)?;
        let provider = Provider::new(signer, args.endpoint, now_ms);
        let provider_object = Object::new_provider(provider, tx_digest);
        store.create_object(provider_object);
        Ok(())
    }

    /// Execute `UpdateProvider`. Always stamps `registered_at_ms`
    /// (heartbeat); endpoint update is optional from the user's
    /// perspective (they may submit the same endpoint to bump the
    /// heartbeat without a logical change).
    fn execute_update(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: UpdateProviderArgs,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        Self::validate_endpoint(&args.endpoint)?;

        // Reject mismatched provider_id ↔ signer pairs. This is the
        // primary auth check: only the address that registered the
        // provider can update it. We compare derived ids rather than
        // reading the object first so the error is the same whether
        // the signer is wrong OR the args.provider_id was forged.
        let expected_id = Provider::derive_id(signer);
        if args.provider_id != expected_id {
            return Err(ExecutionFailureStatus::ProviderCallerMismatch.into());
        }

        let provider_object = store
            .read_object(&args.provider_id)
            .ok_or(ExecutionFailureStatus::ProviderNotFound)?
            .clone();
        let mut provider = provider_object
            .as_provider()
            .ok_or(ExecutionFailureStatus::ProviderNotFound)?;

        // Defense in depth: even though `provider_id` matches the
        // signer-derived id, verify the loaded record's `address`
        // also matches. Catches the (impossible-by-construction but
        // worth-asserting) case where someone forges a Provider
        // object at the right id with the wrong embedded address.
        if provider.address != signer {
            return Err(ExecutionFailureStatus::ProviderCallerMismatch.into());
        }

        let now_ms = Self::read_clock_ts(store)?;
        provider.endpoint = args.endpoint;
        provider.registered_at_ms = now_ms;

        let mut updated = provider_object;
        updated.set_provider_data(&provider);
        store.mutate_input_object(updated);
        Ok(())
    }
}

impl TransactionExecutor for ProviderExecutor {
    fn fee_units(&self, _store: &TemporaryStore, _kind: &TransactionKind) -> u32 {
        // Both ops are cheap shared-object writes.
        1
    }

    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::RegisterProvider(args) => {
                self.execute_register(store, signer, args, tx_digest)
            }
            TransactionKind::UpdateProvider(args) => {
                self.execute_update(store, signer, args, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use protocol_config::Chain;
    use types::CLOCK_OBJECT_ID;
    use types::CLOCK_OBJECT_SHARED_VERSION;
    use types::digests::TransactionDigest;
    use types::object::{ObjectID, ObjectType};
    use types::system_state::FeeParameters;
    use types::temporary_store::TemporaryStore;
    use types::transaction::{
        InputObjectKind, InputObjects, ObjectReadResult, ObjectReadResultKind,
    };

    use super::*;

    /// Helper: build a TemporaryStore with the Clock present at `ts_ms`.
    fn empty_store_with_clock(ts_ms: TimestampMs) -> TemporaryStore {
        let clock_obj = Object::new_clock_with_timestamp_for_testing(ts_ms);
        let inputs = InputObjects::new(vec![ObjectReadResult::new(
            InputObjectKind::SharedObject {
                id: CLOCK_OBJECT_ID,
                initial_shared_version: CLOCK_OBJECT_SHARED_VERSION,
                mutable: false,
            },
            ObjectReadResultKind::Object(clock_obj),
        )]);

        TemporaryStore::new(
            inputs,
            Vec::new(),
            TransactionDigest::default(),
            0,
            FeeParameters { unit_fee: 0 },
            1,
            Chain::Unknown,
        )
    }

    fn empty_store_with_clock_and_provider(
        ts_ms: TimestampMs,
        provider_obj: Object,
    ) -> TemporaryStore {
        let clock_obj = Object::new_clock_with_timestamp_for_testing(ts_ms);
        let provider_id = provider_obj.id();
        let inputs = InputObjects::new(vec![
            ObjectReadResult::new(
                InputObjectKind::SharedObject {
                    id: CLOCK_OBJECT_ID,
                    initial_shared_version: CLOCK_OBJECT_SHARED_VERSION,
                    mutable: false,
                },
                ObjectReadResultKind::Object(clock_obj),
            ),
            ObjectReadResult::new(
                InputObjectKind::SharedObject {
                    id: provider_id,
                    initial_shared_version: types::object::OBJECT_START_VERSION,
                    mutable: true,
                },
                ObjectReadResultKind::Object(provider_obj),
            ),
        ]);
        TemporaryStore::new(
            inputs,
            Vec::new(),
            TransactionDigest::default(),
            0,
            FeeParameters { unit_fee: 0 },
            1,
            Chain::Unknown,
        )
    }

    fn addr(seed: u8) -> SomaAddress {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SomaAddress::new(bytes)
    }

    #[test]
    fn register_creates_provider_at_derived_id() {
        let signer = addr(1);
        let mut store = empty_store_with_clock(1_000);
        let mut exec = ProviderExecutor::new();
        exec.execute_register(
            &mut store,
            signer,
            RegisterProviderArgs { endpoint: "https://prov.example".into() },
            TransactionDigest::default(),
        )
        .expect("register succeeds");

        let id = Provider::derive_id(signer);
        let obj = store.read_object(&id).expect("provider exists");
        assert_eq!(*obj.type_(), ObjectType::Provider);
        let p = obj.as_provider().unwrap();
        assert_eq!(p.address, signer);
        assert_eq!(p.endpoint, "https://prov.example");
        assert_eq!(p.registered_at_ms, 1_000);
    }

    #[test]
    fn register_rejects_duplicate() {
        let signer = addr(2);
        let prov = Provider::new(signer, "https://old.example".into(), 0);
        let prov_obj = Object::new_provider_for_testing(prov);
        let mut store = empty_store_with_clock_and_provider(2_000, prov_obj);

        let mut exec = ProviderExecutor::new();
        let err = exec
            .execute_register(
                &mut store,
                signer,
                RegisterProviderArgs { endpoint: "https://new.example".into() },
                TransactionDigest::default(),
            )
            .expect_err("duplicate must fail");
        assert!(matches!(
            err,
            ExecutionFailureStatus::ProviderAlreadyExists
        ));
    }

    #[test]
    fn register_rejects_empty_endpoint() {
        let signer = addr(3);
        let mut store = empty_store_with_clock(1);
        let mut exec = ProviderExecutor::new();
        let err = exec
            .execute_register(
                &mut store,
                signer,
                RegisterProviderArgs { endpoint: String::new() },
                TransactionDigest::default(),
            )
            .expect_err("empty endpoint must fail");
        assert!(matches!(
            err,
            ExecutionFailureStatus::ProviderInvalidEndpoint { .. }
        ));
    }

    #[test]
    fn register_rejects_oversized_endpoint() {
        let signer = addr(4);
        let mut store = empty_store_with_clock(1);
        let mut exec = ProviderExecutor::new();
        let too_long = "x".repeat(PROVIDER_ENDPOINT_MAX_LEN + 1);
        let err = exec
            .execute_register(
                &mut store,
                signer,
                RegisterProviderArgs { endpoint: too_long },
                TransactionDigest::default(),
            )
            .expect_err("oversize endpoint must fail");
        assert!(matches!(
            err,
            ExecutionFailureStatus::ProviderInvalidEndpoint { .. }
        ));
    }

    #[test]
    fn update_bumps_heartbeat_and_endpoint() {
        let signer = addr(5);
        let prov = Provider::new(signer, "https://old.example".into(), 100);
        let prov_obj = Object::new_provider_for_testing(prov);
        let provider_id = Provider::derive_id(signer);
        let mut store = empty_store_with_clock_and_provider(5_000, prov_obj);

        let mut exec = ProviderExecutor::new();
        exec.execute_update(
            &mut store,
            signer,
            UpdateProviderArgs {
                provider_id,
                endpoint: "https://new.example".into(),
            },
            TransactionDigest::default(),
        )
        .expect("update succeeds");

        let obj = store.read_object(&provider_id).unwrap();
        let p = obj.as_provider().unwrap();
        assert_eq!(p.endpoint, "https://new.example");
        assert_eq!(p.registered_at_ms, 5_000);
        assert_eq!(p.address, signer, "address never changes");
    }

    #[test]
    fn update_heartbeat_with_same_endpoint_succeeds() {
        let signer = addr(6);
        let prov = Provider::new(signer, "https://stable.example".into(), 100);
        let prov_obj = Object::new_provider_for_testing(prov);
        let provider_id = Provider::derive_id(signer);
        let mut store = empty_store_with_clock_and_provider(9_999, prov_obj);

        let mut exec = ProviderExecutor::new();
        exec.execute_update(
            &mut store,
            signer,
            UpdateProviderArgs {
                provider_id,
                endpoint: "https://stable.example".into(),
            },
            TransactionDigest::default(),
        )
        .expect("heartbeat-only update succeeds");

        let p = store.read_object(&provider_id).unwrap().as_provider().unwrap();
        assert_eq!(p.registered_at_ms, 9_999);
    }

    #[test]
    fn update_rejects_wrong_signer() {
        let owner = addr(7);
        let attacker = addr(8);
        let prov = Provider::new(owner, "https://prov.example".into(), 0);
        let prov_obj = Object::new_provider_for_testing(prov);
        let provider_id = Provider::derive_id(owner);
        let mut store = empty_store_with_clock_and_provider(1, prov_obj);

        let mut exec = ProviderExecutor::new();
        let err = exec
            .execute_update(
                &mut store,
                attacker,
                UpdateProviderArgs {
                    provider_id,
                    endpoint: "https://hijacked.example".into(),
                },
                TransactionDigest::default(),
            )
            .expect_err("wrong signer must fail");
        assert!(matches!(
            err,
            ExecutionFailureStatus::ProviderCallerMismatch
        ));
    }

    #[test]
    fn update_rejects_unregistered() {
        let signer = addr(9);
        let mut store = empty_store_with_clock(1);
        let provider_id = Provider::derive_id(signer);

        let mut exec = ProviderExecutor::new();
        let err = exec
            .execute_update(
                &mut store,
                signer,
                UpdateProviderArgs {
                    provider_id,
                    endpoint: "https://prov.example".into(),
                },
                TransactionDigest::default(),
            )
            .expect_err("not-registered must fail");
        assert!(matches!(
            err,
            ExecutionFailureStatus::ProviderNotFound
        ));
    }
}
