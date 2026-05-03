// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use protocol_config::ProtocolConfig;
use tracing::{error, info, instrument};
use types::base::{SequenceNumber, SomaAddress};
use types::config::transaction_deny_config::TransactionDenyConfig;
use types::crypto::GenericSignature;
use types::error::{SomaError, SomaResult};
use types::object::{Object, ObjectID, ObjectRef, Owner, Version};
use types::transaction::{
    CheckedInputObjects, InputObjectKind, InputObjects, ObjectReadResult, ObjectReadResultKind,
    ReceivingObjectReadResult, ReceivingObjects, TransactionData, TransactionKind,
    VerifiedExecutableTransaction,
};

trait IntoChecked {
    fn into_checked(self) -> CheckedInputObjects;
}

impl IntoChecked for InputObjects {
    fn into_checked(self) -> CheckedInputObjects {
        CheckedInputObjects::new_with_checked_transaction_inputs(self)
    }
}

#[instrument(level = "trace", skip_all)]
pub fn check_transaction_input(
    protocol_config: &ProtocolConfig,
    transaction: &TransactionData,
    input_objects: InputObjects,
    receiving_objects: &ReceivingObjects,
) -> SomaResult<CheckedInputObjects> {
    check_transaction_input_inner(protocol_config, transaction, &input_objects, &[])?;
    check_receiving_objects(&input_objects, receiving_objects)?;

    Ok(input_objects.into_checked())
}

// Since the purpose of this function is to audit certified transactions,
// the checks here should be a strict subset of the checks in check_transaction_input().
// For checks not performed in this function but in check_transaction_input(),
// we should add a comment calling out the difference.
#[instrument(level = "trace", skip_all)]
pub fn check_certificate_input(
    cert: &VerifiedExecutableTransaction,
    input_objects: InputObjects,
    protocol_config: &ProtocolConfig,
) -> SomaResult<CheckedInputObjects> {
    let transaction = cert.data().transaction_data();
    check_transaction_input_inner(protocol_config, transaction, &input_objects, &[])?;
    // NB: We do not check receiving objects when executing. Only at signing time do we check.
    // NB: move verifier is only checked at signing time, not at execution.

    Ok(input_objects.into_checked())
}

// Common checks performed for transactions and certificates.
fn check_transaction_input_inner(
    protocol_config: &ProtocolConfig,
    transaction: &TransactionData,
    input_objects: &InputObjects,
    // Overrides the gas objects in the transaction.
    gas_override: &[ObjectRef],
) -> SomaResult {
    let gas = if gas_override.is_empty() { &transaction.gas() } else { gas_override };

    // Stage 13c: gas is balance-mode for non-system txs — gas
    // comes from the sender's USDC accumulator, validated by
    // `prepare_gas` at execution time. A non-empty `gas_payment`
    // on a non-system tx means the caller is using an obsolete
    // coin-mode path; reject up-front. System txs may carry a
    // non-empty `gas_payment` from older test fixtures; the
    // executor ignores it for them.
    if !gas.is_empty() && !transaction.is_system_tx() {
        return Err(SomaError::GasPaymentError(
            "Stage 13c: gas_payment must be empty (balance-mode gas only)".to_string(),
        ));
    }
    check_objects(transaction, input_objects)?;

    Ok(())
}

fn check_receiving_objects(
    input_objects: &InputObjects,
    receiving_objects: &ReceivingObjects,
) -> Result<(), SomaError> {
    let mut objects_in_txn: HashSet<_> =
        input_objects.object_kinds().map(|x| x.object_id()).collect();

    // Since we're at signing we check that every object reference that we are receiving is the
    // most recent version of that object. If it's been received at the version specified we
    // let it through to allow the transaction to run and fail to unlock any other objects in
    // the transaction. Otherwise, we return an error.
    //
    // If there are any object IDs in common (either between receiving objects and input
    // objects) we return an error.
    for ReceivingObjectReadResult { object_ref: (object_id, version, object_digest), object } in
        receiving_objects.iter()
    {
        if !(*version < Version::MAX) {
            return Err(SomaError::InvalidSequenceNumber);
        }

        let Some(object) = object.as_object() else {
            // object was previously received
            continue;
        };

        if !(object.owner.is_address_owned()
            && object.version() == *version
            && object.digest() == *object_digest)
        {
            // Version mismatch
            if !(object.version() == *version) {
                return Err(SomaError::ObjectVersionUnavailableForConsumption {
                    provided_obj_ref: (*object_id, *version, *object_digest),
                    current_version: object.version(),
                });
            }

            // Digest mismatch
            let expected_digest = object.digest();
            if !(expected_digest == *object_digest) {
                return Err(SomaError::InvalidObjectDigest {
                    object_id: *object_id,
                    expected_digest,
                });
            }

            match object.owner {
                Owner::AddressOwner(_) => {
                    debug_assert!(
                        false,
                        "Receiving object {:?} is invalid but we expect it should be valid. {:?}",
                        (*object_id, *version, *object_id),
                        object
                    );
                    error!(
                        "Receiving object {:?} is invalid but we expect it should be valid. {:?}",
                        (*object_id, *version, *object_id),
                        object
                    );
                    // We should never get here, but if for some reason we do just default to
                    // object not found and reject signing the transaction.
                    return Err(SomaError::ObjectNotFound {
                        object_id: *object_id,
                        version: Some(*version),
                    });
                }

                Owner::Shared { .. } => {
                    return Err(SomaError::NotSharedObjectError);
                }
                Owner::Immutable => {
                    return Err(SomaError::MutableParameterExpected { object_id: *object_id });
                }
                Owner::Accumulator { .. } => {
                    // Stage 14a: accumulator objects are system-managed
                    // and cannot be the target of a `Receiving` capability.
                    return Err(SomaError::NotOwnedObjectError);
                }
            };
        }

        if !(!objects_in_txn.contains(object_id)) {
            return Err(SomaError::DuplicateObjectRefInput);
        }

        objects_in_txn.insert(*object_id);
    }
    Ok(())
}

/// Check all the objects used in the transaction against the database, and ensure
/// that they are all the correct version and number.
#[instrument(level = "trace", skip_all)]
fn check_objects(transaction: &TransactionData, objects: &InputObjects) -> SomaResult<()> {
    // We require that mutable objects cannot show up more than once.
    let mut used_objects: HashSet<SomaAddress> = HashSet::new();
    for object in objects.iter() {
        if object.is_mutable() {
            if !(used_objects.insert(object.id().into())) {
                return Err(SomaError::MutableObjectUsedMoreThanOnce { object_id: object.id() });
            }
        }
    }

    // Stage 7: stateless balance-mode txs (e.g. BalanceTransfer) have
    // no owned inputs and no gas coin — empty `InputObjects` is the
    // correct shape, not an arity violation. The replay-protection
    // path (`is_replay_protected()` + `executed_transaction_digests`)
    // covers the safety hole that owned-object versioning used to
    // close.
    let is_balance_mode = transaction.gas().is_empty();
    if !(transaction.is_genesis_tx() || transaction.is_system_tx() || is_balance_mode)
        && objects.is_empty()
    {
        return Err(SomaError::ObjectInputArityViolation);
    }

    let gas_coins: HashSet<ObjectID> =
        HashSet::from_iter(transaction.gas().iter().map(|obj_ref| obj_ref.0));
    for object in objects.iter() {
        let input_object_kind = object.input_object_kind;

        match &object.object {
            ObjectReadResultKind::Object(object) => {
                // For Gas Object, we check the object is owned by gas owner
                let owner_address = transaction.sender();
                // Check if the object contents match the type of lock we need for
                // this object.
                let system_transaction = transaction.is_system_tx();
                check_one_object(&owner_address, input_object_kind, object, system_transaction)?;
            }
            // We skip checking a removed consensus object because it no longer exists.
            ObjectReadResultKind::DeletedSharedObject(_, _) => (),
            // We skip checking shared objects from cancelled transactions since we are not reading it.
            ObjectReadResultKind::CancelledTransactionSharedObject(_) => (),
        }
    }

    Ok(())
}

/// Check one object against a reference
fn check_one_object(
    owner: &SomaAddress,
    object_kind: InputObjectKind,
    object: &Object,
    system_transaction: bool,
) -> SomaResult {
    match object_kind {
        InputObjectKind::ImmOrOwnedObject((object_id, sequence_number, object_digest)) => {
            if !(sequence_number < Version::MAX) {
                return Err(SomaError::InvalidSequenceNumber);
            }

            // This is an invariant - we just load the object with the given ID and version.
            assert_eq!(
                object.version(),
                sequence_number,
                "The fetched object version {:?} does not match the requested version {:?}, object id: {}",
                object.version(),
                sequence_number,
                object.id(),
            );

            // Check the digest matches - user could give a mismatched ObjectDigest
            let expected_digest = object.digest();
            if !(expected_digest == object_digest) {
                return Err(SomaError::InvalidObjectDigest { object_id, expected_digest });
            }

            match object.owner {
                Owner::Immutable => {
                    // Nothing else to check for Immutable.
                }
                Owner::AddressOwner(actual_owner) => {
                    // Check the owner is correct.
                    if !(owner == &actual_owner) {
                        return Err(SomaError::IncorrectUserSignature {
                            error: format!(
                                "Object {object_id:?} is owned by account address {actual_owner:?}, but given owner/signer address is {owner:?}"
                            ),
                        });
                    }
                }

                Owner::Shared { .. } => {
                    // This object is a mutable consensus object. However the transaction
                    // specifies it as an owned object. This is inconsistent.
                    return Err(SomaError::NotOwnedObjectError);
                }
                Owner::Accumulator { .. } => {
                    // Stage 14a: accumulator objects can never be
                    // declared as `ImmOrOwnedObject` inputs by user
                    // transactions. Only privileged executors load
                    // them, and they enter the input set via a
                    // distinct path.
                    return Err(SomaError::NotOwnedObjectError);
                }
            };
        }

        InputObjectKind::SharedObject {
            id: object_id,
            initial_shared_version: input_initial_shared_version,
            ..
        } => {
            if !(object.version() < Version::MAX) {
                return Err(SomaError::InvalidSequenceNumber);
            }

            match &object.owner {
                Owner::AddressOwner(_) | Owner::Immutable | Owner::Accumulator { .. } => {
                    // When someone locks an object as shared it must be shared already.
                    return Err(SomaError::NotSharedObjectError);
                }
                Owner::Shared { initial_shared_version: actual_initial_shared_version } => {
                    if !(input_initial_shared_version == *actual_initial_shared_version) {
                        return Err(SomaError::SharedObjectStartingVersionMismatch);
                    }
                }
            }
        }
    };
    Ok(())
}

macro_rules! deny_if_true {
    ($cond:expr, $msg:expr) => {
        if ($cond) {
            return Err(SomaError::TransactionDenied { error: $msg.to_string() });
        }
    };
}

/// Check that the provided transaction is allowed to be signed according to the
/// deny config.
pub fn check_transaction_for_signing(
    tx_data: &TransactionData,
    tx_signatures: &[GenericSignature],
    input_object_kinds: &[InputObjectKind],
    receiving_objects: &[ObjectRef],
    filter_config: &TransactionDenyConfig,
) -> SomaResult {
    check_disabled_features(filter_config, tx_data, tx_signatures)?;

    check_signers(filter_config, tx_data)?;

    check_input_objects(filter_config, input_object_kinds)?;

    check_receiving_objects_deny(filter_config, receiving_objects)?;

    Ok(())
}

fn check_receiving_objects_deny(
    filter_config: &TransactionDenyConfig,
    receiving_objects: &[ObjectRef],
) -> SomaResult {
    deny_if_true!(
        filter_config.receiving_objects_disabled() && !receiving_objects.is_empty(),
        "Receiving objects is temporarily disabled".to_string()
    );
    for (id, _, _) in receiving_objects {
        deny_if_true!(
            filter_config.get_object_deny_set().contains(id),
            format!("Access to object {:?} is temporarily disabled", id)
        );
    }
    Ok(())
}

fn check_disabled_features(
    filter_config: &TransactionDenyConfig,
    tx_data: &TransactionData,
    tx_signatures: &[GenericSignature],
) -> SomaResult {
    deny_if_true!(
        filter_config.user_transaction_disabled(),
        "Transaction signing is temporarily disabled"
    );

    Ok(())
}

fn check_signers(filter_config: &TransactionDenyConfig, tx_data: &TransactionData) -> SomaResult {
    let deny_map = filter_config.get_address_deny_set();
    if deny_map.is_empty() {
        return Ok(());
    }
    for signer in tx_data.signers() {
        deny_if_true!(
            deny_map.contains(&signer),
            format!("Access to account address {:?} is temporarily disabled", signer)
        );
    }
    Ok(())
}

fn check_input_objects(
    filter_config: &TransactionDenyConfig,
    input_object_kinds: &[InputObjectKind],
) -> SomaResult {
    let deny_map = filter_config.get_object_deny_set();
    let shared_object_disabled = filter_config.shared_object_disabled();
    if deny_map.is_empty() && !shared_object_disabled {
        // No need to iterate through the input objects if no relevant policy is set.
        return Ok(());
    }
    for input_object_kind in input_object_kinds {
        let id = input_object_kind.object_id();
        deny_if_true!(
            deny_map.contains(&id),
            format!("Access to input object {:?} is temporarily disabled", id)
        );
        deny_if_true!(
            shared_object_disabled && input_object_kind.is_shared_object(),
            "Usage of shared object in transactions is temporarily disabled"
        );
    }
    Ok(())
}
