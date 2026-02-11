use std::{
    collections::{BTreeMap, HashSet},
    fmt,
};

use crate::base::ExecutionDigests;
use object_change::{EffectsObjectChange, IDOperation, ObjectIn, ObjectOut};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    base::SomaAddress,
    committee::{Committee, EpochId},
    consensus::block::BlockRef,
    crypto::{
        AuthoritySignInfo, AuthoritySignInfoTrait, AuthorityStrongQuorumSignInfo, EmptySignInfo,
        default_hash,
    },
    digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest},
    envelope::{Envelope, Message, TrustedEnvelope, VerifiedEnvelope},
    error::{SomaError, SomaResult},
    intent::{Intent, IntentScope},
    object::{
        OBJECT_START_VERSION, ObjectID, ObjectRef, ObjectType, Owner, Version, VersionDigest,
    },
    storage::WriteKind,
    temporary_store::SharedInput,
    tensor::SomaTensor,
    tx_fee::TransactionFee,
};

pub mod object_change;

/// # TransactionEffects
///
/// The response from processing a transaction or a certified transaction. This structure
/// contains all information about the outcome of transaction execution, including execution
/// status, object changes, and dependencies.
///
/// ## Purpose
/// TransactionEffects serves as the authoritative record of all state changes resulting from
/// a transaction. It captures the complete set of objects created, modified, or deleted,
/// as well as the transaction's execution status and dependencies.
///
/// ## Lifecycle
/// 1. Created by the transaction executor after processing a transaction
/// 2. Signed by authorities to create SignedTransactionEffects
/// 3. Aggregated into CertifiedTransactionEffects when enough signatures are collected
/// 4. Used to update the global state and notify clients of transaction outcome
///
/// ## Thread Safety
/// This structure is immutable after creation and can be safely shared across threads.
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct TransactionEffects {
    /// The status of the execution (success or failure with reason)
    pub status: ExecutionStatus,

    /// The epoch when this transaction was executed
    pub executed_epoch: EpochId,

    pub transaction_fee: TransactionFee,

    /// The transaction digest that uniquely identifies the transaction
    pub transaction_digest: TransactionDigest,

    /// The updated gas object reference, as an index into the `changed_objects` vector.
    /// Having a dedicated field for convenient access.
    /// System transaction that don't require gas will leave this as None.
    pub gas_object_index: Option<u32>,

    /// The set of transaction digests this transaction depends on
    /// These are transactions that must be executed before this one
    pub dependencies: Vec<TransactionDigest>,

    /// The version number assigned to all written objects by this transaction
    /// All objects modified by a transaction receive the same version number
    pub version: Version,

    /// Objects whose state are changed in the object store
    /// This includes created, modified, and deleted objects
    pub changed_objects: Vec<(ObjectID, EffectsObjectChange)>,

    /// Shared objects that are not mutated in this transaction
    /// Unlike owned objects, read-only shared objects' versions are not committed in the transaction,
    /// and in order for a node to catch up and execute it without consensus sequencing,
    /// the version needs to be committed in the effects.
    pub unchanged_shared_objects: Vec<(ObjectID, UnchangedSharedKind)>,
}

impl TransactionEffectsAPI for TransactionEffects {
    fn status(&self) -> &ExecutionStatus {
        &self.status
    }

    fn into_status(self) -> ExecutionStatus {
        self.status
    }

    fn executed_epoch(&self) -> EpochId {
        self.executed_epoch
    }

    fn transaction_fee(&self) -> &TransactionFee {
        &self.transaction_fee
    }

    fn gas_object(&self) -> (ObjectRef, Owner) {
        if let Some(gas_object_index) = self.gas_object_index {
            let entry = &self.changed_objects[gas_object_index as usize];
            match &entry.1.output_state {
                ObjectOut::ObjectWrite((digest, owner)) => {
                    ((entry.0, self.version, *digest), owner.clone())
                }
                _ => panic!("Gas object must be an ObjectWrite in changed_objects"),
            }
        } else {
            (
                (ObjectID::ZERO, Version::default(), ObjectDigest::MIN),
                Owner::AddressOwner(SomaAddress::default()),
            )
        }
    }

    fn modified_at_versions(&self) -> Vec<(ObjectID, Version)> {
        self.changed_objects
            .iter()
            .filter_map(|(id, change)| {
                if let ObjectIn::Exist(((version, _digest), _owner)) = &change.input_state {
                    Some((*id, *version))
                } else {
                    None
                }
            })
            .collect()
    }

    fn transaction_digest(&self) -> &TransactionDigest {
        &self.transaction_digest
    }

    fn transaction_digest_owned(&self) -> TransactionDigest {
        self.transaction_digest
    }

    fn old_object_metadata(&self) -> Vec<(ObjectRef, Owner)> {
        self.changed_objects
            .iter()
            .filter_map(|(id, change)| {
                if let ObjectIn::Exist(((version, digest), owner)) = &change.input_state {
                    Some(((*id, *version, *digest), owner.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    fn created(&self) -> Vec<(ObjectRef, Owner)> {
        self.changed_objects
            .iter()
            .filter_map(|(id, change)| {
                match (&change.input_state, &change.output_state, &change.id_operation) {
                    (
                        ObjectIn::NotExist,
                        ObjectOut::ObjectWrite((digest, owner)),
                        IDOperation::Created,
                    ) => Some(((*id, self.version, *digest), owner.clone())),
                    _ => None,
                }
            })
            .collect()
    }

    fn mutated(&self) -> Vec<(ObjectRef, Owner)> {
        self.changed_objects
            .iter()
            .filter_map(|(id, change)| match (&change.input_state, &change.output_state) {
                (ObjectIn::Exist(_), ObjectOut::ObjectWrite((digest, owner))) => {
                    Some(((*id, self.version, *digest), owner.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// Return an iterator of mutated objects, but excluding the gas object.
    fn mutated_excluding_gas(&self) -> Vec<(ObjectRef, Owner)> {
        self.mutated().into_iter().filter(|o| o != &self.gas_object()).collect()
    }

    fn deleted(&self) -> Vec<ObjectRef> {
        self.changed_objects
            .iter()
            .filter_map(|(id, change)| {
                match (&change.input_state, &change.output_state, &change.id_operation) {
                    (ObjectIn::Exist(_), ObjectOut::NotExist, IDOperation::Deleted) => {
                        Some((*id, self.version, ObjectDigest::OBJECT_DIGEST_DELETED))
                    }
                    _ => None,
                }
            })
            .collect()
    }

    fn version(&self) -> Version {
        self.version
    }

    fn input_shared_objects(&self) -> Vec<InputSharedObject> {
        self.changed_objects
            .iter()
            .filter_map(|(id, change)| match &change.input_state {
                ObjectIn::Exist(((version, digest), Owner::Shared { .. })) => {
                    Some(InputSharedObject::Mutate((*id, *version, *digest)))
                }
                _ => None,
            })
            .chain(self.unchanged_shared_objects.iter().map(|(id, change_kind)| {
                match change_kind {
                    UnchangedSharedKind::ReadOnlyRoot((version, digest)) => {
                        InputSharedObject::ReadOnly((*id, *version, *digest))
                    }
                    UnchangedSharedKind::MutateDeleted(seqno) => {
                        InputSharedObject::MutateDeleted(*id, *seqno)
                    }
                    UnchangedSharedKind::ReadDeleted(seqno) => {
                        InputSharedObject::ReadDeleted(*id, *seqno)
                    }
                    UnchangedSharedKind::Cancelled(seqno) => {
                        InputSharedObject::Cancelled(*id, *seqno)
                    }
                }
            }))
            .collect()
    }

    fn dependencies(&self) -> &[TransactionDigest] {
        &self.dependencies
    }

    fn unchanged_shared_objects(&self) -> Vec<(ObjectID, UnchangedSharedKind)> {
        self.unchanged_shared_objects.clone()
    }

    fn transaction_digest_mut_for_testing(&mut self) -> &mut TransactionDigest {
        &mut self.transaction_digest
    }

    fn dependencies_mut_for_testing(&mut self) -> &mut Vec<TransactionDigest> {
        &mut self.dependencies
    }

    fn unsafe_add_input_shared_object_for_testing(&mut self, kind: InputSharedObject) {
        match kind {
            InputSharedObject::Mutate(obj_ref) => self.changed_objects.push((
                obj_ref.0,
                EffectsObjectChange {
                    input_state: ObjectIn::Exist((
                        (obj_ref.1, obj_ref.2),
                        Owner::Shared { initial_shared_version: OBJECT_START_VERSION },
                    )),
                    output_state: ObjectOut::ObjectWrite((
                        obj_ref.2,
                        Owner::Shared { initial_shared_version: obj_ref.1 },
                    )),
                    id_operation: IDOperation::None,
                },
            )),
            InputSharedObject::ReadOnly(obj_ref) => self
                .unchanged_shared_objects
                .push((obj_ref.0, UnchangedSharedKind::ReadOnlyRoot((obj_ref.1, obj_ref.2)))),
            InputSharedObject::ReadDeleted(obj_id, seqno) => self
                .unchanged_shared_objects
                .push((obj_id, UnchangedSharedKind::ReadDeleted(seqno))),
            InputSharedObject::MutateDeleted(obj_id, seqno) => self
                .unchanged_shared_objects
                .push((obj_id, UnchangedSharedKind::MutateDeleted(seqno))),
            InputSharedObject::Cancelled(obj_id, seqno) => {
                self.unchanged_shared_objects.push((obj_id, UnchangedSharedKind::Cancelled(seqno)))
            }
        }
    }

    fn written(&self) -> Vec<ObjectRef> {
        self.changed_objects
            .iter()
            .filter_map(|(id, change)| match (&change.output_state, &change.id_operation) {
                (ObjectOut::NotExist, IDOperation::Deleted) => {
                    Some((*id, self.version, ObjectDigest::OBJECT_DIGEST_DELETED))
                }
                (ObjectOut::NotExist, IDOperation::None) => {
                    Some((*id, self.version, ObjectDigest::OBJECT_DIGEST_WRAPPED))
                }
                (ObjectOut::ObjectWrite((d, _)), _) => Some((*id, self.version, *d)),
                _ => None,
            })
            .collect()
    }

    fn object_changes(&self) -> Vec<ObjectChange> {
        self.changed_objects
            .iter()
            .map(|(id, change)| {
                let input_version_digest = match &change.input_state {
                    ObjectIn::NotExist => None,
                    ObjectIn::Exist((vd, _)) => Some(*vd),
                };

                let output_version_digest = match &change.output_state {
                    ObjectOut::NotExist => None,
                    ObjectOut::ObjectWrite((d, _)) => Some((self.version, *d)),
                };

                ObjectChange {
                    id: *id,

                    input_version: input_version_digest.map(|k| k.0),
                    input_digest: input_version_digest.map(|k| k.1),

                    output_version: output_version_digest.map(|k| k.0),
                    output_digest: output_version_digest.map(|k| k.1),

                    id_operation: change.id_operation,
                }
            })
            .collect()
    }
}

impl TransactionEffects {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        status: ExecutionStatus,
        executed_epoch: EpochId,
        shared_objects: Vec<SharedInput>,
        transaction_digest: TransactionDigest,
        version: Version,
        changed_objects: BTreeMap<ObjectID, EffectsObjectChange>,
        dependencies: Vec<TransactionDigest>,
        transaction_fee: TransactionFee,
        gas_object: Option<ObjectID>,
    ) -> Self {
        let unchanged_shared_objects = shared_objects
            .into_iter()
            .filter_map(|shared_input| match shared_input {
                SharedInput::Existing((id, version, digest)) => {
                    if changed_objects.contains_key(&id) {
                        None
                    } else {
                        Some((id, UnchangedSharedKind::ReadOnlyRoot((version, digest))))
                    }
                }
                SharedInput::Deleted((id, version, mutable, _)) => {
                    debug_assert!(!changed_objects.contains_key(&id));
                    if mutable {
                        Some((id, UnchangedSharedKind::MutateDeleted(version)))
                    } else {
                        Some((id, UnchangedSharedKind::ReadDeleted(version)))
                    }
                }
                SharedInput::Cancelled((id, version)) => {
                    debug_assert!(!changed_objects.contains_key(&id));
                    Some((id, UnchangedSharedKind::Cancelled(version)))
                }
            })
            .collect();
        let changed_objects: Vec<_> = changed_objects.into_iter().collect();

        let gas_object_index = gas_object
            .map(|gas_id| changed_objects.iter().position(|(id, _)| id == &gas_id).unwrap() as u32);

        let result = Self {
            status,
            executed_epoch,
            transaction_digest,
            gas_object_index,
            version,
            changed_objects,
            unchanged_shared_objects,
            dependencies,
            transaction_fee,
        };
        #[cfg(debug_assertions)]
        result.check_invariant();

        result
    }

    pub fn execution_digests(&self) -> ExecutionDigests {
        ExecutionDigests { transaction: *self.transaction_digest(), effects: self.digest() }
    }

    /// This function demonstrates what's the invariant of the effects.
    /// It also documents the semantics of different combinations in object changes.
    #[cfg(debug_assertions)]
    fn check_invariant(&self) {
        let mut unique_ids = HashSet::new();
        for (id, change) in &self.changed_objects {
            assert!(unique_ids.insert(*id));
            match (&change.input_state, &change.output_state, &change.id_operation) {
                (ObjectIn::NotExist, ObjectOut::NotExist, IDOperation::Created) => {
                    // created and then wrapped Move object.
                }
                (ObjectIn::NotExist, ObjectOut::NotExist, IDOperation::Deleted) => {
                    // unwrapped and then deleted Move object.
                }
                (ObjectIn::NotExist, ObjectOut::ObjectWrite((_, owner)), IDOperation::None) => {
                    // unwrapped  object.
                    // It's not allowed to make an object shared after unwrapping.
                    assert!(!owner.is_shared());
                }
                (ObjectIn::NotExist, ObjectOut::ObjectWrite(..), IDOperation::Created) => {
                    // created object.
                }
                (
                    ObjectIn::Exist(((old_version, _), old_owner)),
                    ObjectOut::NotExist,
                    IDOperation::None,
                ) => {
                    // wrapped.
                    assert!(old_version.value() < self.version.value());
                    assert!(
                        !old_owner.is_shared() && !old_owner.is_immutable(),
                        "Cannot wrap shared or immutable object"
                    );
                }
                (
                    ObjectIn::Exist(((old_version, _), old_owner)),
                    ObjectOut::NotExist,
                    IDOperation::Deleted,
                ) => {
                    // deleted.
                    assert!(old_version.value() < self.version.value());
                    assert!(!old_owner.is_immutable(), "Cannot delete immutable object");
                }
                (
                    ObjectIn::Exist(((old_version, old_digest), old_owner)),
                    ObjectOut::ObjectWrite((new_digest, new_owner)),
                    IDOperation::None,
                ) => {
                    // mutated.
                    assert!(old_version.value() < self.version.value());
                    assert_ne!(old_digest, new_digest);
                    assert!(!old_owner.is_immutable(), "Cannot mutate immutable object");
                    if old_owner.is_shared() {
                        assert!(new_owner.is_shared(), "Cannot un-share an object");
                    } else {
                        assert!(!new_owner.is_shared(), "Cannot share an existing object");
                    }
                }
                _ => {
                    panic!("Impossible object change: {:?}, {:?}", id, change);
                }
            }
        }

        for (id, _) in &self.unchanged_shared_objects {
            assert!(unique_ids.insert(*id), "Duplicate object id: {:?}\n{:#?}", id, self);
        }
    }

    /// Return an iterator that iterates through all changed objects, including mutated,
    /// created and unwrapped objects. In other words, all objects that still exist
    /// in the object state after this transaction.
    /// It doesn't include deleted objects.
    pub fn all_changed_objects(&self) -> Vec<(ObjectRef, Owner, WriteKind)> {
        self.mutated()
            .into_iter()
            .map(|(r, o)| (r, o, WriteKind::Mutate))
            .chain(self.created().into_iter().map(|(r, o)| (r, o, WriteKind::Create)))
            .collect()
    }

    /// Return all objects that existed in the state prior to the transaction
    /// but no longer exist in the state after the transaction.
    pub fn all_removed_objects(&self) -> Vec<ObjectRef> {
        self.deleted()
    }

    /// Returns all objects that will become a tombstone after this transaction.
    /// This includes deleted, unwrapped_then_deleted and wrapped objects.
    pub fn all_tombstones(&self) -> Vec<(ObjectID, Version)> {
        self.deleted().into_iter().map(|obj_ref| (obj_ref.0, obj_ref.1)).collect()
    }
}

impl Default for TransactionEffects {
    fn default() -> Self {
        Self {
            status: ExecutionStatus::Success,
            executed_epoch: 0,
            transaction_digest: TransactionDigest::default(),
            version: Version::default(),
            changed_objects: vec![],
            dependencies: vec![],
            unchanged_shared_objects: vec![],
            transaction_fee: TransactionFee::default(),
            gas_object_index: None,
        }
    }
}

impl Message for TransactionEffects {
    type DigestType = TransactionEffectsDigest;
    const SCOPE: IntentScope = IntentScope::TransactionData;

    fn digest(&self) -> Self::DigestType {
        TransactionEffectsDigest::new(default_hash(self))
    }
}

pub trait TransactionEffectsAPI {
    fn status(&self) -> &ExecutionStatus;
    fn into_status(self) -> ExecutionStatus;
    fn executed_epoch(&self) -> EpochId;
    fn modified_at_versions(&self) -> Vec<(ObjectID, Version)>;
    fn transaction_digest(&self) -> &TransactionDigest;
    fn transaction_digest_owned(&self) -> TransactionDigest;

    /// The version assigned to all output objects
    fn version(&self) -> Version;

    /// Metadata of objects prior to modification. This includes any object that exists in the
    /// store prior to this transaction and is modified in this transaction.
    /// It includes objects that are mutated and deleted.
    fn old_object_metadata(&self) -> Vec<(ObjectRef, Owner)>;
    /// Returns the list of sequenced shared objects used in the input.
    /// This is needed in effects because in transaction we only have object ID
    /// for shared objects. Their version and digest can only be figured out after sequencing.
    /// Also provides the use kind to indicate whether the object was mutated or read-only.
    /// TODO: Rename this function to indicate sequencing requirement.
    fn input_shared_objects(&self) -> Vec<InputSharedObject>;
    fn created(&self) -> Vec<(ObjectRef, Owner)>;
    fn mutated(&self) -> Vec<(ObjectRef, Owner)>;
    fn deleted(&self) -> Vec<ObjectRef>;
    fn written(&self) -> Vec<ObjectRef>;

    fn dependencies(&self) -> &[TransactionDigest];

    fn deleted_mutably_accessed_shared_objects(&self) -> Vec<ObjectID> {
        self.input_shared_objects()
            .into_iter()
            .filter_map(|kind| match kind {
                InputSharedObject::MutateDeleted(id, _) => Some(id),
                InputSharedObject::Mutate(..)
                | InputSharedObject::ReadOnly(..)
                | InputSharedObject::ReadDeleted(..)
                | InputSharedObject::Cancelled(..) => None,
            })
            .collect()
    }

    /// Returns all root shared objects (i.e. not child object) that are read-only in the transaction.
    fn unchanged_shared_objects(&self) -> Vec<(ObjectID, UnchangedSharedKind)>;

    fn transaction_digest_mut_for_testing(&mut self) -> &mut TransactionDigest;
    fn dependencies_mut_for_testing(&mut self) -> &mut Vec<TransactionDigest>;
    fn unsafe_add_input_shared_object_for_testing(&mut self, kind: InputSharedObject);

    fn transaction_fee(&self) -> &TransactionFee;
    fn mutated_excluding_gas(&self) -> Vec<(ObjectRef, Owner)>;

    fn gas_object(&self) -> (ObjectRef, Owner);

    fn object_changes(&self) -> Vec<ObjectChange>;
}

pub type TransactionEffectsEnvelope<S> = Envelope<TransactionEffects, S>;
pub type UnsignedTransactionEffects = TransactionEffectsEnvelope<EmptySignInfo>;
pub type SignedTransactionEffects = TransactionEffectsEnvelope<AuthoritySignInfo>;
pub type CertifiedTransactionEffects = TransactionEffectsEnvelope<AuthorityStrongQuorumSignInfo>;

pub type TrustedSignedTransactionEffects = TrustedEnvelope<TransactionEffects, AuthoritySignInfo>;
pub type VerifiedTransactionEffectsEnvelope<S> = VerifiedEnvelope<TransactionEffects, S>;
pub type VerifiedSignedTransactionEffects = VerifiedTransactionEffectsEnvelope<AuthoritySignInfo>;
pub type VerifiedCertifiedTransactionEffects =
    VerifiedTransactionEffectsEnvelope<AuthorityStrongQuorumSignInfo>;

impl CertifiedTransactionEffects {
    pub fn verify_authority_signatures(&self, committee: &Committee) -> SomaResult {
        self.auth_sig().verify_secure(self.data(), Intent::soma_transaction(), committee)
    }

    pub fn verify(self, committee: &Committee) -> SomaResult<VerifiedCertifiedTransactionEffects> {
        self.verify_authority_signatures(committee)?;
        Ok(VerifiedCertifiedTransactionEffects::new_from_verified(self))
    }
}

/// # ExecutionStatus
///
/// Represents the outcome of transaction execution - either success or failure with error details.
///
/// ## Purpose
/// Provides a clear indication of whether a transaction executed successfully or failed,
/// and if it failed, the specific reason for the failure.
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Transaction executed successfully
    Success,

    /// Transaction execution failed
    Failure {
        /// The specific error that caused the failure
        error: ExecutionFailureStatus,
    },
}

impl ExecutionStatus {
    pub fn new_failure(error: ExecutionFailureStatus) -> ExecutionStatus {
        ExecutionStatus::Failure { error }
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, ExecutionStatus::Success)
    }

    pub fn is_err(&self) -> bool {
        matches!(self, ExecutionStatus::Failure { .. })
    }

    pub fn unwrap(&self) {
        match self {
            ExecutionStatus::Success => {}
            ExecutionStatus::Failure { .. } => {
                panic!("Unable to unwrap() on {:?}", self);
            }
        }
    }

    pub fn unwrap_err(self) -> ExecutionFailureStatus {
        match self {
            ExecutionStatus::Success => {
                panic!("Unable to unwrap() on {:?}", self);
            }
            ExecutionStatus::Failure { error } => error,
        }
    }
}

/// # ExecutionFailureStatus
///
/// Detailed error types that can occur during transaction execution.
///
/// ## Purpose
/// Provides specific error information when a transaction fails, allowing clients
/// to understand exactly why their transaction was not executed successfully.
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize, Error)]
pub enum ExecutionFailureStatus {
    //
    // General transaction errors
    //
    /// Transaction ran out of gas before completion
    #[error("Insufficient Gas.")]
    InsufficientGas,
    #[error(
        "Invalid owner for object {object_id}. Expected: {expected_owner}, Actual: \
         {actual_owner:?}"
    )]
    InvalidOwnership {
        object_id: ObjectID,
        expected_owner: SomaAddress,
        actual_owner: Option<SomaAddress>,
    },
    #[error("Object not found.")]
    ObjectNotFound { object_id: ObjectID },
    #[error(
        "Invalid object type for object {object_id}. Expected: {expected_type:?}, Actual: \
         {actual_type:?}"
    )]
    InvalidObjectType { object_id: ObjectID, expected_type: ObjectType, actual_type: ObjectType },
    /// Error when the transaction type does not match what is expected
    #[error("The transaction type does not match the expected type")]
    InvalidTransactionType,
    #[error("Invalid arguments passed into transaction: {reason}")]
    InvalidArguments { reason: String },

    //
    // Validator errors
    //
    /// Error when attempting to add a validator that already exists
    #[error("Cannot add validator that is already active or pending")]
    DuplicateValidator,

    /// Error when trying to remove a validator that doesn't exist
    #[error("Cannot remove validator that is not active")]
    NotAValidator,

    /// Error when trying to remove a validator that was already removed
    #[error("Cannot remove validator that is already removed")]
    ValidatorAlreadyRemoved,
    /// Error when advancing to an unexpected epoch
    #[error("Advanced to wrong epoch")]
    AdvancedToWrongEpoch,

    //
    // Model errors
    //
    #[error("Model not found.")]
    ModelNotFound,

    #[error("Sender is not the model owner.")]
    NotModelOwner,

    #[error("Model is not in active state.")]
    ModelNotActive,

    #[error("Model is not in pending (committed) state.")]
    ModelNotPending,

    #[error("Model is already inactive.")]
    ModelAlreadyInactive,

    #[error("Model reveal epoch mismatch: must reveal in the epoch after commit.")]
    ModelRevealEpochMismatch,

    #[error("Model weights URL commitment mismatch.")]
    ModelWeightsUrlMismatch,

    #[error("Model has no pending update.")]
    ModelNoPendingUpdate,

    #[error("Model architecture version mismatch.")]
    ModelArchitectureVersionMismatch,

    #[error("Model commission rate exceeds maximum (10000 bps).")]
    ModelCommissionRateTooHigh,

    #[error("Model minimum stake requirement not met.")]
    ModelMinStakeNotMet,

    //
    // Target errors
    //
    #[error("No active models available for target generation.")]
    NoActiveModels,

    #[error("Target not found.")]
    TargetNotFound,

    #[error("Target is not open for submissions.")]
    TargetNotOpen,

    #[error("Target expired: generation_epoch={generation_epoch}, current_epoch={current_epoch}")]
    TargetExpired { generation_epoch: EpochId, current_epoch: EpochId },

    #[error("Target is not filled.")]
    TargetNotFilled,

    #[error("Challenge window still open: fill_epoch={fill_epoch}, current_epoch={current_epoch}")]
    ChallengeWindowOpen { fill_epoch: EpochId, current_epoch: EpochId },

    #[error("Target rewards have already been claimed.")]
    TargetAlreadyClaimed,

    //
    // Submission errors
    //
    #[error("Model {model_id} is not assigned to target {target_id}.")]
    ModelNotInTarget { model_id: ObjectID, target_id: ObjectID },

    #[error("Embedding dimension mismatch: expected {expected}, got {actual}.")]
    EmbeddingDimensionMismatch { expected: u64, actual: u64 },

    #[error("Distance score {score} exceeds threshold {threshold}.")]
    DistanceExceedsThreshold { score: SomaTensor, threshold: SomaTensor },

    #[error("Insufficient bond: required {required}, provided {provided}.")]
    InsufficientBond { required: u64, provided: u64 },

    #[error("Data size {size} bytes exceeds maximum allowed {max_size} bytes.")]
    DataExceedsMaxSize { size: u64, max_size: u64 },

    #[error("Insufficient emission pool balance.")]
    InsufficientEmissionBalance,

    //
    // Challenge errors
    //
    #[error(
        "Challenge window has closed: fill_epoch={fill_epoch}, current_epoch={current_epoch}"
    )]
    ChallengeWindowClosed {
        fill_epoch: EpochId,
        current_epoch: EpochId,
    },

    #[error("Insufficient challenger bond: required {required}, provided {provided}.")]
    InsufficientChallengerBond { required: u64, provided: u64 },

    #[error("Challenge not found: {challenge_id}")]
    ChallengeNotFound { challenge_id: ObjectID },

    #[error("Challenge is not pending: {challenge_id}")]
    ChallengeNotPending { challenge_id: ObjectID },

    #[error("Challenge already exists for this target (only first challenger wins)")]
    ChallengeAlreadyExists,

    #[error(
        "Challenge expired: challenge_epoch={challenge_epoch}, current_epoch={current_epoch}"
    )]
    ChallengeExpired {
        challenge_epoch: EpochId,
        current_epoch: EpochId,
    },

    #[error("Invalid challenge result: challenge_id mismatch")]
    InvalidChallengeResult,

    #[error("Invalid challenge quorum: signature verification failed")]
    InvalidChallengeQuorum,

    //
    // Coin errors
    //
    /// Account doesn't have enough coins to complete the operation
    #[error("Insufficient coin balance for operation.")]
    InsufficientCoinBalance,

    /// The operation would cause a coin balance to exceed the maximum value
    #[error("The coin balance overflows u64")]
    CoinBalanceOverflow,

    //
    // Validator / Staking errors
    //
    #[error("Validator not found.")]
    ValidatorNotFound,

    #[error("Staking Pool not found.")]
    StakingPoolNotFound,

    #[error("Validator cannot report oneself.")]
    CannotReportOneself,

    #[error("Report record cannot be undone if not reported.")]
    ReportRecordNotFound,

    #[error(
        "Certificate cannot be executed due to a dependency on a deleted shared object or an object that was transferred out of consensus"
    )]
    InputObjectDeleted,

    #[error("Certificate is on the deny list")]
    CertificateDenied,

    #[error("Certificate is cancelled due to congestion on shared object")]
    ExecutionCancelledDueToSharedObjectCongestion, //{ congested_objects: CongestedObjects },

    //
    // Post-execution errors
    //
    /// Generic Soma error that wraps other error types
    #[error("Soma Error {0}")]
    SomaError(SomaError),
}

#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct CongestedObjects(pub Vec<ObjectID>);

impl fmt::Display for CongestedObjects {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        for obj in &self.0 {
            write!(f, "{}, ", obj)?;
        }
        Ok(())
    }
}

/// # InputSharedObject
///
/// Represents different ways a shared object can be accessed as input to a transaction.
///
/// ## Purpose
/// Tracks how shared objects are used in transactions, distinguishing between read-only
/// and mutable access, as well as handling special cases like deleted objects.
///
/// ## Usage
/// Used to properly sequence transactions that access the same shared objects and
/// to ensure correct handling of shared object versions.
#[derive(Eq, PartialEq, Clone, Debug)]
pub enum InputSharedObject {
    /// A shared object that is mutated by the transaction
    Mutate(ObjectRef),

    /// A shared object that is only read by the transaction
    ReadOnly(ObjectRef),

    /// A deleted shared object that is read by the transaction
    ReadDeleted(ObjectID, Version),

    /// A deleted shared object that appears as mutable in the transaction
    MutateDeleted(ObjectID, Version),

    /// A shared object in a cancelled transaction
    Cancelled(ObjectID, Version),
}

impl InputSharedObject {
    pub fn id_and_version(&self) -> (ObjectID, Version) {
        let oref = self.object_ref();
        (oref.0, oref.1)
    }

    pub fn object_ref(&self) -> ObjectRef {
        match self {
            InputSharedObject::Mutate(oref) | InputSharedObject::ReadOnly(oref) => *oref,
            InputSharedObject::ReadDeleted(id, version)
            | InputSharedObject::MutateDeleted(id, version) => {
                (*id, *version, ObjectDigest::OBJECT_DIGEST_DELETED)
            }
            InputSharedObject::Cancelled(id, version) => {
                (*id, *version, ObjectDigest::OBJECT_DIGEST_CANCELLED)
            }
        }
    }
}

/// # UnchangedSharedKind
///
/// Represents different types of shared objects that are not modified by a transaction.
///
/// ## Purpose
/// Tracks shared objects that are accessed but not changed by a transaction,
/// including special cases like deleted objects and cancelled transactions.
///
/// ## Usage
/// Used to properly track shared object access patterns and ensure correct
/// sequencing of transactions that access the same shared objects.
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum UnchangedSharedKind {
    /// Read-only shared objects from the input
    /// We don't really need ObjectDigest for protocol correctness,
    /// but it makes it easier to verify untrusted reads
    ReadOnlyRoot(VersionDigest),

    /// Deleted shared objects that appear mutably/owned in the input
    MutateDeleted(Version),

    /// Deleted shared objects that appear as read-only in the input
    ReadDeleted(Version),

    /// Shared objects in cancelled transaction
    /// The sequence number embeds cancellation reason
    Cancelled(Version),
    // /// Read of a per-epoch config object that should remain the same during an epoch
    // PerEpochConfig,
}

#[derive(Clone, Debug)]
pub struct ObjectChange {
    pub id: ObjectID,
    pub input_version: Option<Version>,
    pub input_digest: Option<ObjectDigest>,
    pub output_version: Option<Version>,
    pub output_digest: Option<ObjectDigest>,
    pub id_operation: IDOperation,
}
