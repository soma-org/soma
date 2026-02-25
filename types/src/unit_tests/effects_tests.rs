// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::base::SomaAddress;
use crate::digests::*;
use crate::effects::object_change::*;
use crate::effects::*;
use crate::envelope::Message;
use crate::object::*;
use crate::tx_fee::TransactionFee;
#[allow(unused_imports)]
use std::collections::BTreeMap;

/// Helper to build a simple TransactionEffects with specified changed_objects.
fn make_effects(
    changed_objects: Vec<(ObjectID, EffectsObjectChange)>,
    gas_object_index: Option<u32>,
) -> TransactionEffects {
    TransactionEffects::V1(TransactionEffectsV1 {
        status: ExecutionStatus::Success,
        executed_epoch: 0,
        transaction_digest: TransactionDigest::random(),
        version: Version::from_u64(5),
        changed_objects,
        dependencies: vec![],
        unchanged_shared_objects: vec![],
        transaction_fee: TransactionFee::default(),
        gas_object_index,
    })
}

#[test]
fn test_effects_bcs_roundtrip() {
    let effects = TransactionEffects::default();
    let bytes = bcs::to_bytes(&effects).unwrap();
    let deserialized: TransactionEffects = bcs::from_bytes(&bytes).unwrap();
    assert_eq!(effects, deserialized);
}

#[test]
fn test_effects_digest_determinism() {
    // Two identical effects should produce the same digest.
    let effects1 = TransactionEffects::default();
    let effects2 = TransactionEffects::default();

    let digest1 = effects1.digest();
    let digest2 = effects2.digest();
    assert_eq!(digest1, digest2);

    // A different effects should produce a different digest.
    let effects3 = TransactionEffects::V1(TransactionEffectsV1 {
        executed_epoch: 99,
        ..TransactionEffectsV1::default()
    });
    let digest3 = effects3.digest();
    assert_ne!(digest1, digest3);
}

#[test]
fn test_effects_created_objects() {
    let obj_id = ObjectID::random();
    let obj_digest = ObjectDigest::random();
    let owner = Owner::AddressOwner(SomaAddress::random());

    let change = EffectsObjectChange {
        input_state: ObjectIn::NotExist,
        output_state: ObjectOut::ObjectWrite((obj_digest, owner.clone())),
        id_operation: IDOperation::Created,
    };

    let effects = make_effects(vec![(obj_id, change)], None);
    let created = effects.created();

    assert_eq!(created.len(), 1);
    assert_eq!(created[0].0.0, obj_id);
    assert_eq!(created[0].0.1, effects.version());
    assert_eq!(created[0].0.2, obj_digest);
    assert_eq!(created[0].1, owner);
}

#[test]
fn test_effects_mutated_objects() {
    let obj_id = ObjectID::random();
    let old_digest = ObjectDigest::random();
    let new_digest = ObjectDigest::random();
    let owner = Owner::AddressOwner(SomaAddress::random());

    let change = EffectsObjectChange {
        input_state: ObjectIn::Exist(((Version::from_u64(3), old_digest), owner.clone())),
        output_state: ObjectOut::ObjectWrite((new_digest, owner.clone())),
        id_operation: IDOperation::None,
    };

    let effects = make_effects(vec![(obj_id, change)], None);
    let mutated = effects.mutated();

    assert_eq!(mutated.len(), 1);
    assert_eq!(mutated[0].0.0, obj_id);
    assert_eq!(mutated[0].0.1, effects.version());
    assert_eq!(mutated[0].0.2, new_digest);
    assert_eq!(mutated[0].1, owner);
}

#[test]
fn test_effects_deleted_objects() {
    let obj_id = ObjectID::random();
    let old_digest = ObjectDigest::random();
    let owner = Owner::AddressOwner(SomaAddress::random());

    let change = EffectsObjectChange {
        input_state: ObjectIn::Exist(((Version::from_u64(2), old_digest), owner)),
        output_state: ObjectOut::NotExist,
        id_operation: IDOperation::Deleted,
    };

    let effects = make_effects(vec![(obj_id, change)], None);
    let deleted = effects.deleted();

    assert_eq!(deleted.len(), 1);
    assert_eq!(deleted[0].0, obj_id);
    assert_eq!(deleted[0].1, effects.version());
    assert_eq!(deleted[0].2, ObjectDigest::OBJECT_DIGEST_DELETED);
}

#[test]
fn test_effects_written_includes_all() {
    let version = Version::from_u64(10);

    // Created object.
    let created_id = ObjectID::random();
    let created_digest = ObjectDigest::random();
    let created_change = EffectsObjectChange {
        input_state: ObjectIn::NotExist,
        output_state: ObjectOut::ObjectWrite((
            created_digest,
            Owner::AddressOwner(SomaAddress::random()),
        )),
        id_operation: IDOperation::Created,
    };

    // Mutated object.
    let mutated_id = ObjectID::random();
    let mutated_old_digest = ObjectDigest::random();
    let mutated_new_digest = ObjectDigest::random();
    let mutated_owner = Owner::AddressOwner(SomaAddress::random());
    let mutated_change = EffectsObjectChange {
        input_state: ObjectIn::Exist((
            (Version::from_u64(5), mutated_old_digest),
            mutated_owner.clone(),
        )),
        output_state: ObjectOut::ObjectWrite((mutated_new_digest, mutated_owner)),
        id_operation: IDOperation::None,
    };

    // Deleted object.
    let deleted_id = ObjectID::random();
    let deleted_old_digest = ObjectDigest::random();
    let deleted_change = EffectsObjectChange {
        input_state: ObjectIn::Exist((
            (Version::from_u64(3), deleted_old_digest),
            Owner::AddressOwner(SomaAddress::random()),
        )),
        output_state: ObjectOut::NotExist,
        id_operation: IDOperation::Deleted,
    };

    let effects = TransactionEffects::V1(TransactionEffectsV1 {
        status: ExecutionStatus::Success,
        executed_epoch: 0,
        transaction_digest: TransactionDigest::random(),
        version,
        changed_objects: vec![
            (created_id, created_change),
            (mutated_id, mutated_change),
            (deleted_id, deleted_change),
        ],
        dependencies: vec![],
        unchanged_shared_objects: vec![],
        transaction_fee: TransactionFee::default(),
        gas_object_index: None,
    });

    let written = effects.written();
    // written() should include created, mutated, and deleted objects.
    assert_eq!(written.len(), 3);

    let written_ids: Vec<ObjectID> = written.iter().map(|r| r.0).collect();
    assert!(written_ids.contains(&created_id));
    assert!(written_ids.contains(&mutated_id));
    assert!(written_ids.contains(&deleted_id));
}

#[test]
fn test_effects_gas_object() {
    let gas_id = ObjectID::random();
    let gas_digest = ObjectDigest::random();
    let gas_owner = Owner::AddressOwner(SomaAddress::random());
    let version = Version::from_u64(7);

    let gas_change = EffectsObjectChange {
        input_state: ObjectIn::Exist((
            (Version::from_u64(3), ObjectDigest::random()),
            gas_owner.clone(),
        )),
        output_state: ObjectOut::ObjectWrite((gas_digest, gas_owner.clone())),
        id_operation: IDOperation::None,
    };

    let effects = TransactionEffects::V1(TransactionEffectsV1 {
        status: ExecutionStatus::Success,
        executed_epoch: 0,
        transaction_digest: TransactionDigest::random(),
        version,
        changed_objects: vec![(gas_id, gas_change)],
        dependencies: vec![],
        unchanged_shared_objects: vec![],
        transaction_fee: TransactionFee::default(),
        gas_object_index: Some(0),
    });

    let (gas_ref, owner) = effects.gas_object();
    assert_eq!(gas_ref.0, gas_id);
    assert_eq!(gas_ref.1, version);
    assert_eq!(gas_ref.2, gas_digest);
    assert_eq!(owner, gas_owner);
}

#[test]
fn test_effects_version_numbers() {
    let version = Version::from_u64(42);

    let id1 = ObjectID::random();
    let id2 = ObjectID::random();

    let change1 = EffectsObjectChange {
        input_state: ObjectIn::NotExist,
        output_state: ObjectOut::ObjectWrite((
            ObjectDigest::random(),
            Owner::AddressOwner(SomaAddress::random()),
        )),
        id_operation: IDOperation::Created,
    };

    let change2 = EffectsObjectChange {
        input_state: ObjectIn::Exist((
            (Version::from_u64(10), ObjectDigest::random()),
            Owner::AddressOwner(SomaAddress::random()),
        )),
        output_state: ObjectOut::ObjectWrite((
            ObjectDigest::random(),
            Owner::AddressOwner(SomaAddress::random()),
        )),
        id_operation: IDOperation::None,
    };

    let effects = TransactionEffects::V1(TransactionEffectsV1 {
        status: ExecutionStatus::Success,
        executed_epoch: 0,
        transaction_digest: TransactionDigest::random(),
        version,
        changed_objects: vec![(id1, change1), (id2, change2)],
        dependencies: vec![],
        unchanged_shared_objects: vec![],
        transaction_fee: TransactionFee::default(),
        gas_object_index: None,
    });

    // All written objects should have the effects version.
    let written = effects.written();
    for obj_ref in &written {
        assert_eq!(obj_ref.1, version, "All written objects must have the effects version");
    }

    // Also check via created() and mutated().
    for (obj_ref, _owner) in effects.created() {
        assert_eq!(obj_ref.1, version);
    }
    for (obj_ref, _owner) in effects.mutated() {
        assert_eq!(obj_ref.1, version);
    }
}

#[test]
fn test_execution_status_success() {
    let status = ExecutionStatus::Success;
    assert!(status.is_ok());
    assert!(!status.is_err());
    // unwrap should not panic on success.
    status.unwrap();
}

#[test]
fn test_execution_status_failure() {
    let status = ExecutionStatus::new_failure(ExecutionFailureStatus::InsufficientGas);
    assert!(!status.is_ok());
    assert!(status.is_err());

    let err = status.unwrap_err();
    assert_eq!(err, ExecutionFailureStatus::InsufficientGas);

    // Test other failure variants.
    let status2 = ExecutionStatus::Failure {
        error: ExecutionFailureStatus::ObjectNotFound { object_id: ObjectID::random() },
    };
    assert!(status2.is_err());

    let status3 = ExecutionStatus::new_failure(ExecutionFailureStatus::InsufficientCoinBalance);
    assert!(status3.is_err());
    assert_eq!(status3.unwrap_err(), ExecutionFailureStatus::InsufficientCoinBalance);
}

#[test]
fn test_object_change_variants() {
    // ObjectIn::NotExist
    let obj_in_not_exist = ObjectIn::NotExist;
    assert_eq!(obj_in_not_exist, ObjectIn::NotExist);

    // ObjectIn::Exist
    let version = Version::from_u64(3);
    let digest = ObjectDigest::random();
    let owner = Owner::AddressOwner(SomaAddress::random());
    let obj_in_exist = ObjectIn::Exist(((version, digest), owner.clone()));
    match &obj_in_exist {
        ObjectIn::Exist(((v, d), o)) => {
            assert_eq!(*v, version);
            assert_eq!(*d, digest);
            assert_eq!(*o, owner);
        }
        _ => panic!("Expected ObjectIn::Exist"),
    }

    // ObjectOut::NotExist
    let obj_out_not_exist = ObjectOut::NotExist;
    assert_eq!(obj_out_not_exist, ObjectOut::NotExist);

    // ObjectOut::ObjectWrite
    let out_digest = ObjectDigest::random();
    let out_owner = Owner::Immutable;
    let obj_out_write = ObjectOut::ObjectWrite((out_digest, out_owner.clone()));
    match &obj_out_write {
        ObjectOut::ObjectWrite((d, o)) => {
            assert_eq!(*d, out_digest);
            assert_eq!(*o, out_owner);
        }
        _ => panic!("Expected ObjectOut::ObjectWrite"),
    }

    // IDOperation variants
    assert_eq!(IDOperation::None, IDOperation::None);
    assert_eq!(IDOperation::Created, IDOperation::Created);
    assert_eq!(IDOperation::Deleted, IDOperation::Deleted);
    assert_ne!(IDOperation::Created, IDOperation::Deleted);
    assert_ne!(IDOperation::None, IDOperation::Created);

    // EffectsObjectChange construction
    let change = EffectsObjectChange {
        input_state: ObjectIn::NotExist,
        output_state: ObjectOut::ObjectWrite((out_digest, Owner::Immutable)),
        id_operation: IDOperation::Created,
    };
    assert_eq!(change.input_state, ObjectIn::NotExist);
    assert_eq!(change.id_operation, IDOperation::Created);
}

#[test]
fn test_effects_gas_object_none() {
    // When gas_object_index is None, gas_object() should return a default/zero reference.
    let effects = TransactionEffects::default();
    let (gas_ref, owner) = effects.gas_object();
    assert_eq!(gas_ref.0, ObjectID::ZERO);
    assert_eq!(gas_ref.1, Version::default());
    assert_eq!(gas_ref.2, ObjectDigest::MIN);
    assert_eq!(owner, Owner::AddressOwner(SomaAddress::default()));
}

#[test]
fn test_effects_bcs_roundtrip_with_objects() {
    // A more complex BCS roundtrip with actual changed objects.
    let obj_id = ObjectID::random();
    let obj_digest = ObjectDigest::random();
    let owner = Owner::AddressOwner(SomaAddress::random());

    let change = EffectsObjectChange {
        input_state: ObjectIn::NotExist,
        output_state: ObjectOut::ObjectWrite((obj_digest, owner.clone())),
        id_operation: IDOperation::Created,
    };

    let effects = make_effects(vec![(obj_id, change)], None);

    let bytes = bcs::to_bytes(&effects).unwrap();
    let deserialized: TransactionEffects = bcs::from_bytes(&bytes).unwrap();
    assert_eq!(effects, deserialized);

    // Digest should also match after roundtrip.
    assert_eq!(effects.digest(), deserialized.digest());
}
