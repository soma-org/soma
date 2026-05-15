// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Executor for per-(provider, model) `Offering` transactions.
//!
//! Three ops:
//!
//! | Op                  | Caller                       | Purpose                                                 |
//! |---------------------|------------------------------|---------------------------------------------------------|
//! | RegisterOffering    | anyone (becomes the provider)| Creates an Offering object at derive_id(signer,model_id)|
//! | UpdateOffering      | offering.provider            | Changes prices and SLA bounds                           |
//! | DeactivateOffering  | offering.provider            | Marks offering inactive (no new channels can open)      |
//!
//! `model_id` is always validated against the protocol-config
//! [`protocol_config::ModelRegistry`] at the version recorded in
//! `SystemState`. Models not in the registry — or registered but
//! `active=false` — cannot have new offerings registered against them.
//!
//! Channels are unaffected by offering mutations: they snapshot the
//! offering's price + SLA at `OpenChannel` time. See
//! `execution::channel::execute_open` for the snapshot logic.

use protocol_config::{ModelRegistry, ProtocolVersion};
use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::Object;
use types::offering::Offering;
use types::system_state::SystemState;
use types::temporary_store::TemporaryStore;
use types::transaction::{
    DeactivateOfferingArgs, RegisterOfferingArgs, TransactionKind, UpdateOfferingArgs,
};

use super::TransactionExecutor;

pub struct OfferingExecutor;

impl OfferingExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }

    /// Read the protocol version off SystemState (declared as a
    /// read-only shared input by every offering tx kind).
    fn read_protocol_version(store: &TemporaryStore) -> ExecutionResult<ProtocolVersion> {
        let state_object = store.read_object(&types::SYSTEM_STATE_OBJECT_ID).ok_or(
            ExecutionFailureStatus::ObjectNotFound { object_id: types::SYSTEM_STATE_OBJECT_ID },
        )?;
        let state =
            SystemState::deserialize(state_object.as_inner().data.contents()).map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize SystemState: {}",
                    e
                )))
            })?;
        Ok(ProtocolVersion::new(types::system_state::SystemStateTrait::protocol_version(&state)))
    }

    /// Read the wall-clock Clock timestamp for `updated_at_ms` on the
    /// offering object.
    fn read_clock_ts(store: &TemporaryStore) -> ExecutionResult<u64> {
        store.read_clock_timestamp_ms().ok_or(ExecutionFailureStatus::ChannelClockMissing)
    }

    /// Look up `model_id` in the ModelRegistry at the executor's
    /// protocol version. Rejects unknown or deactivated entries.
    fn validate_model_id(store: &TemporaryStore, model_id: &str) -> ExecutionResult<()> {
        let version = Self::read_protocol_version(store)?;
        let registry = ModelRegistry::for_version(version);
        match registry.get(model_id) {
            Some(_) => Ok(()),
            None => {
                Err(ExecutionFailureStatus::OfferingUnknownModel { model_id: model_id.to_string() }
                    .into())
            }
        }
    }

    fn execute_register(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: RegisterOfferingArgs,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        Self::validate_model_id(store, &args.model_id)?;
        let updated_at_ms = Self::read_clock_ts(store)?;

        let offering_id = Offering::derive_id(signer, &args.model_id);
        if store.read_object(&offering_id).is_some() {
            return Err(ExecutionFailureStatus::OfferingAlreadyExists.into());
        }

        let offering = Offering::new(
            signer,
            args.model_id,
            args.prompt_micros_per_1k,
            args.completion_micros_per_1k,
            args.cache_read_micros_per_1k,
            args.cache_write_micros_per_1k,
            args.request_micros,
            args.ttft_bound_ms,
            args.ttot_bound_ms,
            updated_at_ms,
        );

        let offering_object = Object::new_offering(offering, tx_digest);
        store.create_object(offering_object);
        Ok(())
    }

    fn execute_update(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: UpdateOfferingArgs,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        Self::validate_model_id(store, &args.model_id)?;
        let updated_at_ms = Self::read_clock_ts(store)?;

        // Auth + id consistency: only the provider who registered the
        // offering can update it, and `args.offering_id` must equal the
        // canonical derive_id for (signer, model_id).
        let expected_id = Offering::derive_id(signer, &args.model_id);
        if args.offering_id != expected_id {
            return Err(ExecutionFailureStatus::OfferingCallerMismatch.into());
        }

        let offering_object = store
            .read_object(&args.offering_id)
            .ok_or(ExecutionFailureStatus::OfferingNotFound)?
            .clone();
        let mut offering =
            offering_object.as_offering().ok_or(ExecutionFailureStatus::OfferingNotFound)?;
        if offering.provider() != signer {
            return Err(ExecutionFailureStatus::OfferingCallerMismatch.into());
        }
        if offering.model_id() != args.model_id {
            // Should be impossible since offering_id derives from model_id;
            // catch the (forged-object) corner case.
            return Err(ExecutionFailureStatus::OfferingCallerMismatch.into());
        }

        // Mutate in place — single V1 arm today; future v2 forces this
        // match to be exhaustive.
        let Offering::V1(inner) = &mut offering;
        inner.prompt_micros_per_1k = args.prompt_micros_per_1k;
        inner.completion_micros_per_1k = args.completion_micros_per_1k;
        inner.cache_read_micros_per_1k = args.cache_read_micros_per_1k;
        inner.cache_write_micros_per_1k = args.cache_write_micros_per_1k;
        inner.request_micros = args.request_micros;
        inner.ttft_bound_ms = args.ttft_bound_ms;
        inner.ttot_bound_ms = args.ttot_bound_ms;
        // Re-activating a previously-deactivated row is allowed via
        // Update — providers shouldn't have to dance around protocol
        // states. Deactivation is a one-way for the *current* version.
        inner.active = true;
        inner.updated_at_ms = updated_at_ms;

        let mut updated = offering_object;
        updated.set_offering_data(&offering);
        store.mutate_input_object(updated);
        Ok(())
    }

    fn execute_deactivate(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: DeactivateOfferingArgs,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let updated_at_ms = Self::read_clock_ts(store)?;
        let expected_id = Offering::derive_id(signer, &args.model_id);
        if args.offering_id != expected_id {
            return Err(ExecutionFailureStatus::OfferingCallerMismatch.into());
        }

        let offering_object = store
            .read_object(&args.offering_id)
            .ok_or(ExecutionFailureStatus::OfferingNotFound)?
            .clone();
        let mut offering =
            offering_object.as_offering().ok_or(ExecutionFailureStatus::OfferingNotFound)?;
        if offering.provider() != signer {
            return Err(ExecutionFailureStatus::OfferingCallerMismatch.into());
        }

        let Offering::V1(inner) = &mut offering;
        inner.active = false;
        inner.updated_at_ms = updated_at_ms;

        let mut updated = offering_object;
        updated.set_offering_data(&offering);
        store.mutate_input_object(updated);
        Ok(())
    }
}

impl TransactionExecutor for OfferingExecutor {
    fn fee_units(&self, _store: &TemporaryStore, _kind: &TransactionKind) -> u32 {
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
            TransactionKind::RegisterOffering(args) => {
                self.execute_register(store, signer, args, tx_digest)
            }
            TransactionKind::UpdateOffering(args) => {
                self.execute_update(store, signer, args, tx_digest)
            }
            TransactionKind::DeactivateOffering(args) => {
                self.execute_deactivate(store, signer, args, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use fastcrypto::ed25519::Ed25519KeyPair;
    use protocol_config::Chain;
    use protocol_config::ProtocolConfig;
    use types::CLOCK_OBJECT_ID;
    use types::CLOCK_OBJECT_SHARED_VERSION;
    use types::SYSTEM_STATE_OBJECT_SHARED_VERSION;
    use types::crypto::get_key_pair;
    use types::object::ObjectType;
    use types::system_state::FeeParameters;
    use types::transaction::{
        DeactivateOfferingArgs, InputObjectKind, InputObjects, ObjectReadResult,
        ObjectReadResultKind, RegisterOfferingArgs, UpdateOfferingArgs,
    };

    /// First active entry in the v1 ModelRegistry. Used by the
    /// register tests so we land an offering against a real model.
    const VALID_MODEL: &str = "anthropic/claude-sonnet-4.6";

    fn make_store(clock_ts: u64, offering: Option<Offering>) -> TemporaryStore {
        let mut inputs: Vec<ObjectReadResult> = Vec::new();

        // SystemState (read-only) for protocol version.
        let protocol_config =
            ProtocolConfig::get_for_version(protocol_config::ProtocolVersion::MIN, Chain::Unknown);
        let state = types::system_state::SystemState::create(
            Vec::new(),
            protocol_config.version.as_u64(),
            0,
            &protocol_config,
            0,
            0,
            0,
            0,
            None,
            types::bridge::MarketplaceParameters::default(),
            types::bridge::BridgeCommittee::empty(),
        );
        let state_obj = types::object::Object::new(
            types::object::ObjectData::new_with_id(
                types::SYSTEM_STATE_OBJECT_ID,
                ObjectType::SystemState,
                SYSTEM_STATE_OBJECT_SHARED_VERSION,
                bcs::to_bytes(&state).unwrap(),
            ),
            types::object::Owner::Shared {
                initial_shared_version: SYSTEM_STATE_OBJECT_SHARED_VERSION,
            },
            TransactionDigest::default(),
        );
        inputs.push(ObjectReadResult::new(
            InputObjectKind::SharedObject {
                id: types::SYSTEM_STATE_OBJECT_ID,
                initial_shared_version: SYSTEM_STATE_OBJECT_SHARED_VERSION,
                mutable: false,
            },
            ObjectReadResultKind::Object(state_obj),
        ));

        // Clock for updated_at_ms.
        let clock_obj = types::object::Object::new_clock_with_timestamp_for_testing(clock_ts);
        inputs.push(ObjectReadResult::new(
            InputObjectKind::SharedObject {
                id: CLOCK_OBJECT_ID,
                initial_shared_version: CLOCK_OBJECT_SHARED_VERSION,
                mutable: false,
            },
            ObjectReadResultKind::Object(clock_obj),
        ));

        // Optional preloaded Offering (for update/deactivate).
        if let Some(o) = offering {
            let id = Offering::derive_id(o.provider(), o.model_id());
            let obj = types::object::Object::new_offering_for_testing(o);
            inputs.push(ObjectReadResult::new(
                InputObjectKind::SharedObject {
                    id,
                    initial_shared_version: types::object::OBJECT_START_VERSION,
                    mutable: true,
                },
                ObjectReadResultKind::Object(obj),
            ));
        }

        TemporaryStore::new(
            InputObjects::new(inputs),
            Vec::new(),
            TransactionDigest::default(),
            0,
            FeeParameters { unit_fee: 0 },
            0,
            Chain::Unknown,
        )
    }

    fn provider_addr() -> SomaAddress {
        let (addr, _kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
        addr
    }

    /// Happy path: a provider registers an offering for a model in the
    /// protocol-config registry. Object materializes at the derived
    /// id with the supplied prices and SLA.
    #[test]
    fn register_offering_happy_path() {
        let provider = provider_addr();
        let mut store = make_store(1_700_000_000_000, None);
        let mut exec = OfferingExecutor::new();

        exec.execute_register(
            &mut store,
            provider,
            RegisterOfferingArgs {
                model_id: VALID_MODEL.to_string(),
                prompt_micros_per_1k: 3_000,
                completion_micros_per_1k: 15_000,
                cache_read_micros_per_1k: 300,
                cache_write_micros_per_1k: 3_000,
                request_micros: 0,
                ttft_bound_ms: 1_500,
                ttot_bound_ms: 50,
            },
            TransactionDigest::default(),
        )
        .expect("register succeeds");

        // Find the created Offering object.
        let offering_obj = store
            .execution_results
            .written_objects
            .values()
            .find(|o| *o.type_() == ObjectType::Offering)
            .expect("offering created");
        let o = offering_obj.as_offering().unwrap();
        assert_eq!(o.provider(), provider);
        assert_eq!(o.model_id(), VALID_MODEL);
        assert_eq!(o.prompt_micros_per_1k(), 3_000);
        assert_eq!(o.completion_micros_per_1k(), 15_000);
        assert_eq!(o.ttft_bound_ms(), 1_500);
        assert!(o.active());
    }

    /// Registering against a model_id not in the protocol-config
    /// registry must be rejected.
    #[test]
    fn register_offering_rejects_unknown_model() {
        let provider = provider_addr();
        let mut store = make_store(0, None);
        let mut exec = OfferingExecutor::new();

        let status = exec
            .execute_register(
                &mut store,
                provider,
                RegisterOfferingArgs {
                    model_id: "fake/never-existed".to_string(),
                    prompt_micros_per_1k: 1,
                    completion_micros_per_1k: 1,
                    cache_read_micros_per_1k: 0,
                    cache_write_micros_per_1k: 0,
                    request_micros: 0,
                    ttft_bound_ms: 100,
                    ttot_bound_ms: 10,
                },
                TransactionDigest::default(),
            )
            .expect_err("unknown model_id rejected");
        match status {
            ExecutionFailureStatus::OfferingUnknownModel { model_id } => {
                assert_eq!(model_id, "fake/never-existed");
            }
            other => panic!("unexpected status: {:?}", other),
        }
    }

    /// Duplicate registration for the same (provider, model_id) is
    /// rejected — providers use Update for price changes.
    #[test]
    fn register_offering_rejects_duplicate() {
        let provider = provider_addr();
        let existing = Offering::new(provider, VALID_MODEL.to_string(), 1, 2, 3, 4, 5, 100, 10, 0);
        let mut store = make_store(0, Some(existing));
        let mut exec = OfferingExecutor::new();

        let status = exec
            .execute_register(
                &mut store,
                provider,
                RegisterOfferingArgs {
                    model_id: VALID_MODEL.to_string(),
                    prompt_micros_per_1k: 9,
                    completion_micros_per_1k: 9,
                    cache_read_micros_per_1k: 0,
                    cache_write_micros_per_1k: 0,
                    request_micros: 0,
                    ttft_bound_ms: 100,
                    ttot_bound_ms: 10,
                },
                TransactionDigest::default(),
            )
            .expect_err("duplicate registration rejected");
        assert!(matches!(status, ExecutionFailureStatus::OfferingAlreadyExists));
    }

    /// Update mutates prices + SLA on an existing offering. Caller
    /// must be the provider who registered it.
    #[test]
    fn update_offering_happy_path() {
        let provider = provider_addr();
        let existing = Offering::new(
            provider,
            VALID_MODEL.to_string(),
            1_000,
            5_000,
            0,
            0,
            0,
            2_000,
            100,
            1_700_000_000_000,
        );
        let mut store = make_store(1_700_000_001_000, Some(existing));
        let mut exec = OfferingExecutor::new();
        let offering_id = Offering::derive_id(provider, VALID_MODEL);

        exec.execute_update(
            &mut store,
            provider,
            UpdateOfferingArgs {
                offering_id,
                model_id: VALID_MODEL.to_string(),
                prompt_micros_per_1k: 2_000,
                completion_micros_per_1k: 10_000,
                cache_read_micros_per_1k: 200,
                cache_write_micros_per_1k: 2_000,
                request_micros: 0,
                ttft_bound_ms: 800,
                ttot_bound_ms: 40,
            },
            TransactionDigest::default(),
        )
        .expect("update succeeds");

        let updated = store.read_object(&offering_id).unwrap().as_offering().unwrap();
        assert_eq!(updated.prompt_micros_per_1k(), 2_000);
        assert_eq!(updated.completion_micros_per_1k(), 10_000);
        assert_eq!(updated.ttft_bound_ms(), 800);
        assert_eq!(updated.updated_at_ms(), 1_700_000_001_000);
        assert!(updated.active());
    }

    /// Update from a non-provider signer is rejected (auth check).
    #[test]
    fn update_offering_rejects_caller_mismatch() {
        let provider = provider_addr();
        let imposter = provider_addr();
        assert_ne!(provider, imposter);
        let existing = Offering::new(provider, VALID_MODEL.to_string(), 1, 2, 3, 4, 5, 100, 10, 0);
        let mut store = make_store(0, Some(existing));
        let mut exec = OfferingExecutor::new();
        let offering_id = Offering::derive_id(provider, VALID_MODEL);

        let status = exec
            .execute_update(
                &mut store,
                imposter,
                UpdateOfferingArgs {
                    offering_id,
                    model_id: VALID_MODEL.to_string(),
                    prompt_micros_per_1k: 9,
                    completion_micros_per_1k: 9,
                    cache_read_micros_per_1k: 0,
                    cache_write_micros_per_1k: 0,
                    request_micros: 0,
                    ttft_bound_ms: 100,
                    ttot_bound_ms: 10,
                },
                TransactionDigest::default(),
            )
            .expect_err("imposter rejected");
        assert!(matches!(status, ExecutionFailureStatus::OfferingCallerMismatch));
    }

    /// Deactivate flips `active=false`; the row stays so existing
    /// channels can keep operating against the snapshot.
    #[test]
    fn deactivate_offering_happy_path() {
        let provider = provider_addr();
        let existing = Offering::new(
            provider,
            VALID_MODEL.to_string(),
            1,
            2,
            3,
            4,
            5,
            100,
            10,
            1_700_000_000_000,
        );
        let mut store = make_store(1_700_000_002_000, Some(existing));
        let mut exec = OfferingExecutor::new();
        let offering_id = Offering::derive_id(provider, VALID_MODEL);

        exec.execute_deactivate(
            &mut store,
            provider,
            DeactivateOfferingArgs { offering_id, model_id: VALID_MODEL.to_string() },
            TransactionDigest::default(),
        )
        .expect("deactivate succeeds");

        let after = store.read_object(&offering_id).unwrap().as_offering().unwrap();
        assert!(!after.active());
        assert_eq!(after.updated_at_ms(), 1_700_000_002_000);
    }
}
