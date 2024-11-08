use std::collections::BTreeMap;

use object_change::{EffectsObjectChange, IDOperation, ObjectIn, ObjectOut};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    committee::{Committee, EpochId},
    crypto::{
        default_hash, AuthoritySignInfo, AuthoritySignInfoTrait, AuthorityStrongQuorumSignInfo,
        EmptySignInfo,
    },
    digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest},
    envelope::{Envelope, Message, TrustedEnvelope, VerifiedEnvelope},
    error::{SomaError, SomaResult},
    intent::{Intent, IntentScope},
    object::{ObjectID, ObjectRef, Version},
    storage::WriteKind,
};

pub mod object_change;

/// The response from processing a transaction or a certified transaction
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct TransactionEffects {
    /// The status of the execution
    status: ExecutionStatus,
    /// The epoch when this transaction was executed.
    executed_epoch: EpochId,
    /// The transaction digest
    transaction_digest: TransactionDigest,
    /// Objects whose state are changed in the object store.
    changed_objects: Vec<(ObjectID, EffectsObjectChange)>,
    /// The version number of all the written objects by this transaction.
    pub(crate) version: Version,
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

    fn modified_at_versions(&self) -> Vec<(ObjectID, Version)> {
        self.changed_objects
            .iter()
            .filter_map(|(id, change)| {
                if let ObjectIn::Exist((version, _owner)) = &change.input_state {
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

    fn old_object_metadata(&self) -> Vec<ObjectRef> {
        self.changed_objects
            .iter()
            .filter_map(|(id, change)| {
                if let ObjectIn::Exist((version, digest)) = &change.input_state {
                    Some((*id, *version, *digest))
                } else {
                    None
                }
            })
            .collect()
    }

    fn created(&self) -> Vec<ObjectRef> {
        self.changed_objects
            .iter()
            .filter_map(|(id, change)| {
                match (
                    &change.input_state,
                    &change.output_state,
                    &change.id_operation,
                ) {
                    (ObjectIn::NotExist, ObjectOut::ObjectWrite(digest), IDOperation::Created) => {
                        Some((*id, self.version, *digest))
                    }
                    _ => None,
                }
            })
            .collect()
    }

    fn mutated(&self) -> Vec<ObjectRef> {
        self.changed_objects
            .iter()
            .filter_map(
                |(id, change)| match (&change.input_state, &change.output_state) {
                    (ObjectIn::Exist(_), ObjectOut::ObjectWrite(digest)) => {
                        Some((*id, self.version, *digest))
                    }
                    _ => None,
                },
            )
            .collect()
    }

    fn deleted(&self) -> Vec<ObjectRef> {
        self.changed_objects
            .iter()
            .filter_map(|(id, change)| {
                match (
                    &change.input_state,
                    &change.output_state,
                    &change.id_operation,
                ) {
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
}

impl TransactionEffects {
    pub fn new(
        status: ExecutionStatus,
        executed_epoch: EpochId,
        transaction_digest: TransactionDigest,
        version: Version,
        changed_objects: BTreeMap<ObjectID, EffectsObjectChange>,
    ) -> Self {
        let changed_objects: Vec<_> = changed_objects.into_iter().collect();
        Self {
            status,
            executed_epoch,
            transaction_digest,
            version,
            changed_objects,
        }
    }

    /// Return an iterator that iterates through all changed objects, including mutated,
    /// created and unwrapped objects. In other words, all objects that still exist
    /// in the object state after this transaction.
    /// It doesn't include deleted objects.
    pub fn all_changed_objects(&self) -> Vec<(ObjectRef, WriteKind)> {
        self.mutated()
            .into_iter()
            .map(|r| (r, WriteKind::Mutate))
            .chain(self.created().into_iter().map(|r| (r, WriteKind::Create)))
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
        self.deleted()
            .into_iter()
            .map(|obj_ref| (obj_ref.0, obj_ref.1))
            .collect()
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

    /// The version assigned to all output objects
    fn version(&self) -> Version;

    /// Metadata of objects prior to modification. This includes any object that exists in the
    /// store prior to this transaction and is modified in this transaction.
    /// It includes objects that are mutated and deleted.
    fn old_object_metadata(&self) -> Vec<ObjectRef>;

    fn created(&self) -> Vec<ObjectRef>;
    fn mutated(&self) -> Vec<ObjectRef>;
    fn deleted(&self) -> Vec<ObjectRef>;
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
        self.auth_sig()
            .verify_secure(self.data(), Intent::soma_transaction(), committee)
    }

    pub fn verify(self, committee: &Committee) -> SomaResult<VerifiedCertifiedTransactionEffects> {
        self.verify_authority_signatures(committee)?;
        Ok(VerifiedCertifiedTransactionEffects::new_from_verified(self))
    }
}

#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Success,
    /// Gas used in the failed case, and the error.
    Failure {
        /// The error
        error: ExecutionFailureStatus,
    },
}

impl ExecutionStatus {
    pub fn new_failure(error: ExecutionFailureStatus) -> ExecutionStatus {
        ExecutionStatus::Failure { error }
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, ExecutionStatus::Success { .. })
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
            ExecutionStatus::Success { .. } => {
                panic!("Unable to unwrap() on {:?}", self);
            }
            ExecutionStatus::Failure { error } => error,
        }
    }
}

#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize, Error)]
pub enum ExecutionFailureStatus {
    //
    // General transaction errors
    //
    #[error("Insufficient Gas.")]
    InsufficientGas,

    //
    // Coin errors
    //
    #[error("Insufficient coin balance for operation.")]
    InsufficientCoinBalance,
    #[error("The coin balance overflows u64")]
    CoinBalanceOverflow,

    //
    // Post-execution errors
    //
    // Indicates the effects from the transaction are too large
    #[error(
        "Effects of size {current_size} bytes too large. \
    Limit is {max_size} bytes"
    )]
    EffectsTooLarge { current_size: u64, max_size: u64 },

    #[error("Certificate is cancelled because randomness could not be generated this epoch")]
    ExecutionCancelledDueToRandomnessUnavailable,

    #[error("Soma Error {0}")]
    SomaError(SomaError),
}
