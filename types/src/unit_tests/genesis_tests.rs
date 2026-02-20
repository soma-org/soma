//! Tests for the genesis builder and genesis state construction.
//!
//! These tests verify that the `GenesisBuilder` correctly constructs the initial
//! blockchain state, including system state, validators, token allocations,
//! emission pools, targets, checkpoints, and transaction structure.

use protocol_config::ProtocolVersion;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    config::genesis_config::{
        GenesisCeremonyParameters, GenesisModelConfig, SHANNONS_PER_SOMA,
        TokenAllocation, TokenDistributionSchedule, TokenDistributionScheduleBuilder,
        ValidatorGenesisConfigBuilder,
    },
    effects::{ExecutionStatus, TransactionEffectsAPI},
    envelope::Message,
    genesis_builder::GenesisBuilder,
    object::{ObjectType, Owner},
    system_state::{SystemStateTrait, get_system_state},
    system_state::epoch_start::EpochStartSystemStateTrait,
    transaction::TransactionKind,
};

// ---------------------------------------------------------------------------
// Helper: build N validator configs with a deterministic RNG
// ---------------------------------------------------------------------------

fn make_validator_configs(n: usize) -> Vec<crate::config::genesis_config::ValidatorGenesisConfig> {
    let mut rng = StdRng::from_seed([42u8; 32]);
    (0..n)
        .map(|_| ValidatorGenesisConfigBuilder::new().build(&mut rng))
        .collect()
}

/// Build a simple TokenDistributionSchedule that allocates `per_validator`
/// shannons to each validator (staked) and sends the rest to the emission fund.
fn make_schedule_for_validators(
    configs: &[crate::config::genesis_config::ValidatorGenesisConfig],
    per_validator: u64,
) -> TokenDistributionSchedule {
    let mut builder = TokenDistributionScheduleBuilder::new();
    for config in configs {
        let address = SomaAddress::from(&config.account_key_pair.public());
        builder.add_allocation(TokenAllocation {
            recipient_address: address,
            amount_shannons: per_validator,
            staked_with_validator: Some(address),
            staked_with_model: None,
        });
    }
    builder.build()
}

/// Build a TokenDistributionSchedule with both validator stakes and
/// free (unstaked) coin allocations.
fn make_schedule_with_coins(
    configs: &[crate::config::genesis_config::ValidatorGenesisConfig],
    per_validator_stake: u64,
    coin_recipients: &[(SomaAddress, u64)],
) -> TokenDistributionSchedule {
    let mut builder = TokenDistributionScheduleBuilder::new();
    for config in configs {
        let address = SomaAddress::from(&config.account_key_pair.public());
        builder.add_allocation(TokenAllocation {
            recipient_address: address,
            amount_shannons: per_validator_stake,
            staked_with_validator: Some(address),
            staked_with_model: None,
        });
    }
    for (addr, amount) in coin_recipients {
        builder.add_allocation(TokenAllocation {
            recipient_address: *addr,
            amount_shannons: *amount,
            staked_with_validator: None,
            staked_with_model: None,
        });
    }
    builder.build()
}

/// Fully build genesis (unsigned) from N validators with default parameters.
/// Returns the UnsignedGenesis for inspection.
fn build_unsigned_genesis_with_validators(
    n: usize,
) -> (crate::genesis::UnsignedGenesis, Vec<crate::config::genesis_config::ValidatorGenesisConfig>)
{
    let configs = make_validator_configs(n);
    let schedule = make_schedule_for_validators(&configs, 1_000 * SHANNONS_PER_SOMA);
    let unsigned = GenesisBuilder::new()
        .with_validator_configs(configs.clone())
        .with_token_distribution_schedule(schedule)
        .build_unsigned_genesis();
    (unsigned, configs)
}

// ===========================================================================
// Test 1: SystemState object exists with epoch 0
// ===========================================================================

#[test]
fn test_genesis_creates_system_state() {
    let (unsigned, _configs) = build_unsigned_genesis_with_validators(4);

    // The system state must be retrievable from genesis objects
    let system_state = get_system_state(&unsigned.objects()).expect("SystemState must exist");

    // Epoch must be 0 at genesis
    assert_eq!(system_state.epoch(), 0, "Genesis epoch must be 0");

    // The SYSTEM_STATE_OBJECT_ID must be present among objects
    let ss_obj = unsigned.object(SYSTEM_STATE_OBJECT_ID);
    assert!(ss_obj.is_some(), "SystemState object must exist with well-known ID");
}

// ===========================================================================
// Test 2: All configured validators appear in the committee
// ===========================================================================

#[test]
fn test_genesis_creates_validators() {
    let (unsigned, configs) = build_unsigned_genesis_with_validators(4);

    let system_state = get_system_state(&unsigned.objects()).unwrap();
    assert_eq!(
        system_state.validators().validators.len(),
        configs.len(),
        "Validator count must match configured count"
    );

    // Verify every configured validator address is present
    let validator_addrs: Vec<SomaAddress> = system_state
        .validators()
        .validators
        .iter()
        .map(|v| v.metadata.soma_address)
        .collect();

    for config in &configs {
        let expected_addr = SomaAddress::from(&config.account_key_pair.public());
        assert!(
            validator_addrs.contains(&expected_addr),
            "Validator {} from config must be present in genesis committee",
            expected_addr
        );
    }

    // Each validator should have non-zero voting power (set_voting_power was called)
    for v in &system_state.validators().validators {
        assert!(v.voting_power > 0, "Validator voting power must be set");
    }
}

// ===========================================================================
// Test 3: Token allocations match the distribution schedule
// ===========================================================================

#[test]
fn test_genesis_creates_initial_coins() {
    let configs = make_validator_configs(2);
    let coin_addr = SomaAddress::random();
    let coin_amount = 500 * SHANNONS_PER_SOMA;
    let per_validator = 1_000 * SHANNONS_PER_SOMA;

    let schedule = make_schedule_with_coins(&configs, per_validator, &[(coin_addr, coin_amount)]);

    let unsigned = GenesisBuilder::new()
        .with_validator_configs(configs)
        .with_token_distribution_schedule(schedule)
        .build_unsigned_genesis();

    // Find Coin objects owned by coin_addr
    let coin_objects: Vec<_> = unsigned
        .objects()
        .iter()
        .filter(|o| *o.type_() == ObjectType::Coin && o.owner == Owner::AddressOwner(coin_addr))
        .collect();

    assert!(!coin_objects.is_empty(), "Must have at least one coin object for the recipient");

    // Verify balance
    let total_coin_balance: u64 = coin_objects
        .iter()
        .map(|o| {
            let balance: u64 = bcs::from_bytes(o.as_inner().data.contents()).unwrap();
            balance
        })
        .sum();

    assert_eq!(
        total_coin_balance, coin_amount,
        "Total coin balance must match allocation"
    );
}

// ===========================================================================
// Test 4: Protocol version is set correctly
// ===========================================================================

#[test]
fn test_genesis_protocol_version() {
    let (unsigned, _) = build_unsigned_genesis_with_validators(4);
    let system_state = get_system_state(&unsigned.objects()).unwrap();

    // Default parameters use ProtocolVersion::max()
    assert_eq!(
        system_state.protocol_version(),
        ProtocolVersion::max().as_u64(),
        "Protocol version must match configured version"
    );
}

// ===========================================================================
// Test 5: Builder caching produces identical genesis on repeated calls
// ===========================================================================

#[test]
fn test_genesis_deterministic() {
    let configs = make_validator_configs(4);
    let schedule = make_schedule_for_validators(&configs, 1_000 * SHANNONS_PER_SOMA);

    let mut params = GenesisCeremonyParameters::new();
    params.chain_start_timestamp_ms = 1_700_000_000_000;

    // The genesis builder uses ObjectID::random() internally for staking pool IDs,
    // coin object IDs, etc. Two separate GenesisBuilder instances will produce
    // different random IDs and thus different digests. However, the builder
    // caches the result of build_unsigned_genesis(), so calling it twice on the
    // same instance must return the same result.
    let mut builder = GenesisBuilder::new()
        .with_parameters(params)
        .with_validator_configs(configs)
        .with_token_distribution_schedule(schedule);

    let unsigned_1 = builder.build_unsigned_genesis();
    let unsigned_2 = builder.build_unsigned_genesis();

    // Checkpoint digests must be identical from the cached builder
    assert_eq!(
        unsigned_1.checkpoint().digest(),
        unsigned_2.checkpoint().digest(),
        "Repeated builds from the same builder must produce the same checkpoint digest"
    );

    // Transaction digests must be identical
    assert_eq!(
        unsigned_1.transaction().digest(),
        unsigned_2.transaction().digest(),
        "Repeated builds from the same builder must produce the same transaction digest"
    );

    // The two results must be equal
    assert_eq!(
        unsigned_1, unsigned_2,
        "Repeated calls to build_unsigned_genesis must return identical results"
    );
}

// ===========================================================================
// Test 6: Custom SystemParameters are respected
// ===========================================================================

#[test]
fn test_genesis_builder_custom_parameters() {
    let configs = make_validator_configs(4);
    let schedule = make_schedule_for_validators(&configs, 1_000 * SHANNONS_PER_SOMA);

    let custom_epoch_duration_ms = 12_000; // 12 seconds (very short for testing)
    let custom_emission = 500_000 * SHANNONS_PER_SOMA;
    let custom_timestamp = 1_700_000_000_000u64;

    let mut params = GenesisCeremonyParameters::new();
    params.epoch_duration_ms = custom_epoch_duration_ms;
    params.emission_per_epoch = custom_emission;
    params.chain_start_timestamp_ms = custom_timestamp;

    let unsigned = GenesisBuilder::new()
        .with_parameters(params)
        .with_validator_configs(configs)
        .with_token_distribution_schedule(schedule)
        .build_unsigned_genesis();

    let system_state = get_system_state(&unsigned.objects()).unwrap();

    assert_eq!(
        system_state.epoch_duration_ms(),
        custom_epoch_duration_ms,
        "Custom epoch duration must be reflected"
    );
    assert_eq!(
        system_state.emission_pool().emission_per_epoch, custom_emission,
        "Custom emission per epoch must be reflected"
    );
    assert_eq!(
        system_state.epoch_start_timestamp_ms(), custom_timestamp,
        "Custom chain start timestamp must be reflected"
    );
}

// ===========================================================================
// Test 7: Works with 1, 4, and 7 validator configs
// ===========================================================================

#[test]
fn test_genesis_builder_multiple_validators() {
    for n in [1, 4, 7] {
        let (unsigned, _configs) = build_unsigned_genesis_with_validators(n);
        let system_state = get_system_state(&unsigned.objects()).unwrap();

        assert_eq!(
            system_state.validators().validators.len(),
            n,
            "Genesis with {} validators must produce {} validators in state",
            n,
            n
        );

        // Committee should be extractable
        let committee = system_state.into_epoch_start_state().get_committee();
        assert_eq!(
            committee.num_members(),
            n,
            "Committee must have {} members",
            n
        );
    }
}

// ===========================================================================
// Test 8: EmissionPool is initialized correctly with proper balance
// ===========================================================================

#[test]
fn test_genesis_emission_pool() {
    let configs = make_validator_configs(4);
    let per_validator = 1_000 * SHANNONS_PER_SOMA;
    let schedule = make_schedule_for_validators(&configs, per_validator);

    // The emission fund is: TOTAL_SUPPLY_SHANNONS - sum(allocations)
    let total_allocated: u64 = configs.len() as u64 * per_validator;
    let expected_emission_fund =
        crate::config::genesis_config::TOTAL_SUPPLY_SHANNONS - total_allocated;

    assert_eq!(
        schedule.emission_fund_shannons, expected_emission_fund,
        "Schedule emission fund must equal total supply minus allocations"
    );

    let unsigned = GenesisBuilder::new()
        .with_validator_configs(configs)
        .with_token_distribution_schedule(schedule)
        .build_unsigned_genesis();

    let system_state = get_system_state(&unsigned.objects()).unwrap();

    // The emission pool balance may be slightly less than emission_fund_shannons
    // if targets were generated at genesis (they draw from the pool).
    // But emission_per_epoch should match the default.
    assert!(
        system_state.emission_pool().balance <= expected_emission_fund,
        "Emission pool balance must not exceed the emission fund allocation"
    );
    assert!(
        system_state.emission_pool().is_emitting(),
        "Emission pool should still be emitting after genesis"
    );

    // Verify emission_per_epoch matches the default
    let default_params = GenesisCeremonyParameters::new();
    assert_eq!(
        system_state.emission_pool().emission_per_epoch,
        default_params.emission_per_epoch,
        "Emission per epoch must match the configured value"
    );
}

// ===========================================================================
// Test 9: Seed models produce active targets when configured
// ===========================================================================

#[test]
fn test_genesis_creates_initial_targets() {
    let configs = make_validator_configs(4);
    let per_validator = 1_000 * SHANNONS_PER_SOMA;

    let model_owner = SomaAddress::random();
    let model_id = crate::model::ModelId::random();
    let model_stake = 1 * SHANNONS_PER_SOMA;

    // Build schedule that includes a model stake allocation
    let mut sched_builder = TokenDistributionScheduleBuilder::new();
    for config in &configs {
        let address = SomaAddress::from(&config.account_key_pair.public());
        sched_builder.add_allocation(TokenAllocation {
            recipient_address: address,
            amount_shannons: per_validator,
            staked_with_validator: Some(address),
            staked_with_model: None,
        });
    }
    // Add a model stake allocation
    sched_builder.add_allocation(TokenAllocation {
        recipient_address: model_owner,
        amount_shannons: model_stake,
        staked_with_model: Some(model_id),
        staked_with_validator: None,
    });
    let schedule = sched_builder.build();

    // Create a genesis model config
    use crate::digests::{ModelWeightsCommitment, ModelWeightsUrlCommitment};
    use crate::metadata::{Manifest, ManifestV1, Metadata, MetadataV1};
    use crate::model::ModelWeightsManifest;
    use crate::checksum::Checksum;
    use crate::crypto::{DecryptionKey, DefaultHash};
    use fastcrypto::hash::HashFunction as _;
    use url::Url;

    let url_str = "https://example.com/model/weights";
    let url = Url::parse(url_str).unwrap();
    let metadata = Metadata::V1(MetadataV1::new(Checksum::new_from_hash([1u8; 32]), 1024));
    let manifest = Manifest::V1(ManifestV1::new(url, metadata));
    let weights_manifest = ModelWeightsManifest {
        manifest,
        decryption_key: DecryptionKey::new([0xAA; 32]),
    };

    let url_bytes = url_str.as_bytes();
    let url_hash = {
        let mut hasher = DefaultHash::default();
        hasher.update(url_bytes);
        let h = hasher.finalize();
        let bytes: [u8; 32] = h.as_ref().try_into().unwrap();
        bytes
    };

    let genesis_model = GenesisModelConfig {
        owner: model_owner,
        model_id,
        weights_manifest,
        weights_url_commitment: ModelWeightsUrlCommitment::new(url_hash),
        weights_commitment: ModelWeightsCommitment::new([0xBB; 32]),
        architecture_version: 1,
        commission_rate: 500, // 5%
        initial_stake: model_stake,
    };

    let unsigned = GenesisBuilder::new()
        .with_validator_configs(configs)
        .with_token_distribution_schedule(schedule)
        .with_genesis_models(vec![genesis_model])
        .build_unsigned_genesis();

    let system_state = get_system_state(&unsigned.objects()).unwrap();

    // Model must be registered as active
    assert!(
        system_state.model_registry().active_models.contains_key(&model_id),
        "Genesis model must be active in the model registry"
    );

    // Targets should have been generated since there is at least one active model
    let target_objects: Vec<_> = unsigned
        .objects()
        .iter()
        .filter(|o| *o.type_() == ObjectType::Target)
        .collect();

    // With an active model and a non-empty emission pool, we expect some targets
    assert!(
        !target_objects.is_empty(),
        "Genesis with an active model should produce seed targets"
    );

    // All targets should be shared objects
    for t in &target_objects {
        assert!(
            matches!(t.owner, Owner::Shared { .. }),
            "Target objects must be Shared"
        );
    }
}

// ===========================================================================
// Test 10: Object ownership correctness
// ===========================================================================

#[test]
fn test_genesis_objects_have_correct_owners() {
    let configs = make_validator_configs(4);
    let coin_addr = SomaAddress::random();
    let coin_amount = 500 * SHANNONS_PER_SOMA;

    let schedule = make_schedule_with_coins(
        &configs,
        1_000 * SHANNONS_PER_SOMA,
        &[(coin_addr, coin_amount)],
    );

    let unsigned = GenesisBuilder::new()
        .with_validator_configs(configs)
        .with_token_distribution_schedule(schedule)
        .build_unsigned_genesis();

    // SystemState must be Shared
    let ss_obj = unsigned.object(SYSTEM_STATE_OBJECT_ID).unwrap();
    assert!(
        matches!(ss_obj.owner, Owner::Shared { .. }),
        "SystemState must be a Shared object"
    );

    // Coin objects must be AddressOwner
    for obj in unsigned.objects() {
        match obj.type_() {
            ObjectType::Coin => {
                assert!(
                    matches!(obj.owner, Owner::AddressOwner(_)),
                    "Coin objects must have AddressOwner ownership"
                );
            }
            ObjectType::StakedSoma => {
                assert!(
                    matches!(obj.owner, Owner::AddressOwner(_)),
                    "StakedSoma objects must have AddressOwner ownership"
                );
            }
            ObjectType::SystemState => {
                assert!(
                    matches!(obj.owner, Owner::Shared { .. }),
                    "SystemState must be Shared"
                );
            }
            ObjectType::Target => {
                assert!(
                    matches!(obj.owner, Owner::Shared { .. }),
                    "Target objects must be Shared"
                );
            }
            _ => {}
        }
    }
}

// ===========================================================================
// Test 11: Genesis checkpoint is valid at sequence 0
// ===========================================================================

#[test]
fn test_genesis_checkpoint() {
    let (unsigned, _configs) = build_unsigned_genesis_with_validators(4);

    let checkpoint = unsigned.checkpoint();

    // Epoch must be 0
    assert_eq!(checkpoint.epoch, 0, "Genesis checkpoint epoch must be 0");

    // Sequence number must be 0
    assert_eq!(
        checkpoint.sequence_number, 0,
        "Genesis checkpoint sequence number must be 0"
    );

    // No previous digest (it is the first checkpoint)
    assert!(
        checkpoint.previous_digest.is_none(),
        "Genesis checkpoint must have no previous digest"
    );

    // Content digest must match checkpoint contents
    let contents = unsigned.checkpoint_contents();
    assert_eq!(
        checkpoint.content_digest,
        *contents.digest(),
        "Checkpoint content_digest must match the actual contents digest"
    );

    // network_total_transactions should be > 0 (at least the genesis transaction)
    assert!(
        checkpoint.network_total_transactions > 0,
        "Genesis checkpoint must include at least one transaction"
    );
}

// ===========================================================================
// Test 12: Genesis transaction is correctly constructed as system tx
// ===========================================================================

#[test]
fn test_genesis_transaction() {
    let (unsigned, _configs) = build_unsigned_genesis_with_validators(4);

    let tx = unsigned.transaction();
    let tx_data = tx.data().transaction_data();

    // Must be a genesis transaction
    assert!(
        tx_data.is_genesis_tx(),
        "Genesis transaction must be identified as genesis"
    );
    assert!(
        tx_data.is_system_tx(),
        "Genesis transaction must be a system transaction"
    );

    // Kind should be TransactionKind::Genesis
    assert!(
        matches!(tx_data.kind(), TransactionKind::Genesis(_)),
        "Transaction kind must be Genesis variant"
    );

    // Effects must show success
    let effects = unsigned.effects();
    assert!(
        matches!(effects.status(), ExecutionStatus::Success),
        "Genesis transaction effects must show Success"
    );

    // Effects must have epoch 0
    assert_eq!(
        effects.executed_epoch(), 0,
        "Genesis effects executed_epoch must be 0"
    );

    // Effects transaction_digest must match the transaction
    assert_eq!(
        *effects.transaction_digest(),
        *tx.digest(),
        "Effects digest must match the genesis transaction digest"
    );
}

// ===========================================================================
// Test 13: Signed genesis build (full ceremony with signatures)
// ===========================================================================

#[test]
fn test_genesis_signed_build() {
    let configs = make_validator_configs(4);
    let schedule = make_schedule_for_validators(&configs, 1_000 * SHANNONS_PER_SOMA);

    let mut builder = GenesisBuilder::new()
        .with_validator_configs(configs.clone())
        .with_token_distribution_schedule(schedule);

    // Sign with each validator's authority keypair
    for config in &configs {
        builder = builder.add_validator_signature(&config.key_pair);
    }

    // Build the fully signed genesis
    let genesis = builder.build();

    // Must be able to get the committee
    let committee = genesis.committee().expect("Committee must be extractable");
    assert_eq!(committee.num_members(), 4);

    // Checkpoint should be verifiable
    let verified_checkpoint = genesis.checkpoint();
    assert_eq!(verified_checkpoint.epoch, 0);
    assert_eq!(verified_checkpoint.sequence_number, 0);
}

// ===========================================================================
// Test 14: with_protocol_version sets the correct version
// ===========================================================================

#[test]
fn test_genesis_with_protocol_version() {
    let configs = make_validator_configs(4);
    let schedule = make_schedule_for_validators(&configs, 1_000 * SHANNONS_PER_SOMA);

    let v1 = ProtocolVersion::new(1);

    let unsigned = GenesisBuilder::new()
        .with_protocol_version(v1)
        .with_validator_configs(configs)
        .with_token_distribution_schedule(schedule)
        .build_unsigned_genesis();

    let system_state = get_system_state(&unsigned.objects()).unwrap();
    assert_eq!(
        system_state.protocol_version(),
        1,
        "Protocol version must be 1 when explicitly set"
    );
}

// ===========================================================================
// Test 15: Genesis effects contain all created objects
// ===========================================================================

#[test]
fn test_genesis_effects_created_objects() {
    let (unsigned, _configs) = build_unsigned_genesis_with_validators(4);

    let effects = unsigned.effects();

    // Effects should have created objects
    assert!(
        !effects.created().is_empty(),
        "Genesis effects must have created objects"
    );

    // Every object in the genesis set must appear in the effects' created list
    let created_ids: std::collections::HashSet<_> =
        effects.created().iter().map(|(oref, _owner)| oref.0).collect();

    for obj in unsigned.objects() {
        assert!(
            created_ids.contains(&obj.id()),
            "Object {} must appear in genesis effects created list",
            obj.id()
        );
    }
}

// ===========================================================================
// Test 16: Genesis with no token distribution schedule (minimal)
// ===========================================================================

#[test]
fn test_genesis_no_token_distribution() {
    let configs = make_validator_configs(4);

    // Building without a token distribution schedule should still succeed
    // (emission fund defaults to 0)
    let unsigned = GenesisBuilder::new()
        .with_validator_configs(configs)
        .build_unsigned_genesis();

    let system_state = get_system_state(&unsigned.objects()).unwrap();
    assert_eq!(system_state.epoch(), 0);
    assert_eq!(system_state.validators().validators.len(), 4);

    // Emission pool balance should be 0 since no schedule was provided
    assert_eq!(
        system_state.emission_pool().balance, 0,
        "Emission pool balance must be 0 without token distribution schedule"
    );
}

// ===========================================================================
// Test 17: Staked allocations create StakedSoma objects
// ===========================================================================

#[test]
fn test_genesis_staked_allocations() {
    let configs = make_validator_configs(4);
    let per_validator = 1_000 * SHANNONS_PER_SOMA;
    let schedule = make_schedule_for_validators(&configs, per_validator);

    let unsigned = GenesisBuilder::new()
        .with_validator_configs(configs.clone())
        .with_token_distribution_schedule(schedule)
        .build_unsigned_genesis();

    // There should be StakedSoma objects for each validator's staked allocation
    let staked_objects: Vec<_> = unsigned
        .objects()
        .iter()
        .filter(|o| *o.type_() == ObjectType::StakedSoma)
        .collect();

    assert_eq!(
        staked_objects.len(),
        configs.len(),
        "Must have one StakedSoma object per validator allocation"
    );

    // Each StakedSoma should be owned by an address
    for obj in &staked_objects {
        assert!(
            matches!(obj.owner, Owner::AddressOwner(_)),
            "StakedSoma objects must be address-owned"
        );
    }
}

// ===========================================================================
// Test 18: Genesis BCS serialization roundtrip for UnsignedGenesis
// ===========================================================================

#[test]
fn test_genesis_unsigned_serialization_roundtrip() {
    let (unsigned, _configs) = build_unsigned_genesis_with_validators(4);

    let bytes = bcs::to_bytes(&unsigned).expect("Must serialize UnsignedGenesis");
    let deserialized: crate::genesis::UnsignedGenesis =
        bcs::from_bytes(&bytes).expect("Must deserialize UnsignedGenesis");

    // Re-serialize the deserialized value to compare at the byte level.
    // Direct PartialEq comparison may fail due to cached OnceCell digest
    // state differences (OnceCell fields are skipped during serde but
    // included in derived PartialEq).
    let re_serialized = bcs::to_bytes(&deserialized).expect("Must re-serialize");
    assert_eq!(
        bytes, re_serialized,
        "UnsignedGenesis must survive BCS roundtrip (bytes must be identical)"
    );

    // Also verify the deserialized system state is functionally identical
    let original_system_state = get_system_state(&unsigned.objects()).unwrap();
    let deser_system_state = get_system_state(&deserialized.objects()).unwrap();
    assert_eq!(
        original_system_state.epoch(),
        deser_system_state.epoch(),
        "SystemState epoch must survive roundtrip"
    );
    assert_eq!(
        original_system_state.validators().validators.len(),
        deser_system_state.validators().validators.len(),
        "Validator count must survive roundtrip"
    );
    assert_eq!(
        original_system_state.emission_pool(),
        deser_system_state.emission_pool(),
        "EmissionPool must survive roundtrip"
    );
}
