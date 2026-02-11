use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    iter,
};

use crate::{
    base::FullObjectID,
    challenge::ChallengeId,
    checkpoints::{CheckpointSequenceNumber, CheckpointTimestamp},
    digests::{
        AdditionalConsensusStateDigest, DataCommitment, ModelWeightsCommitment,
        ModelWeightsUrlCommitment, SenderSignedDataDigest,
    },
    model::{ArchitectureVersion, ModelId, ModelWeightsManifest},
    submission::SubmissionManifest,
    target::TargetId,
    tensor::SomaTensor,
};
use fastcrypto::{
    hash::HashFunction,
    traits::{Signer, ToFromBytes},
};
use itertools::{Either, Itertools};
use nonempty::{NonEmpty, nonempty};
use protocol_config::ProtocolVersion;
use serde::{Deserialize, Serialize};
use tracing::trace;

use crate::{
    SYSTEM_STATE_OBJECT_ID, SYSTEM_STATE_OBJECT_SHARED_VERSION,
    base::{AuthorityName, SizeOneVec, SomaAddress},
    committee::{Committee, EpochId},
    consensus::ConsensusCommitPrologue,
    crypto::{
        AuthoritySignInfo, AuthoritySignInfoTrait, AuthoritySignature,
        AuthorityStrongQuorumSignInfo, DefaultHash, Ed25519SomaSignature, EmptySignInfo,
        GenericSignature, Signature, SomaSignatureInner, default_hash,
    },
    digests::{CertificateDigest, ConsensusCommitDigest, TransactionDigest},
    envelope::{Envelope, Message, TrustedEnvelope, VerifiedEnvelope},
    error::{SomaError, SomaResult},
    intent::{Intent, IntentMessage, IntentScope},
    object::{Object, ObjectID, ObjectRef, Owner, Version, VersionDigest},
    temporary_store::SharedInput,
};
use tap::Pipe;

/// # TransactionKind
///
/// Represents the different types of transactions supported by the Soma blockchain.
///
/// ## Purpose
/// Categorizes transactions based on their function and lifecycle within the system,
/// allowing for specialized processing and validation rules for each type.
///
/// ## Variants
/// - `Genesis`: Initial transaction that creates the genesis state
/// - `ConsensusCommitPrologue`: System transaction that records consensus commit information
/// - `StateTransaction`: User-initiated transaction that modifies blockchain state
/// - `EndOfEpochTransaction`: System transaction that handles epoch transitions
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum TransactionKind {
    /// Genesis transaction that initializes the blockchain state
    Genesis(GenesisTransaction),
    /// Records consensus commit information in the blockchain state
    ConsensusCommitPrologue(ConsensusCommitPrologue),
    /// Transaction that changes the epoch, run by each validator at end of epoch
    ChangeEpoch(ChangeEpoch),
    // Validator management transactions
    AddValidator(AddValidatorArgs),
    RemoveValidator(RemoveValidatorArgs),
    // Reporting validators
    ReportValidator {
        reportee: SomaAddress,
    },
    UndoReportValidator {
        reportee: SomaAddress,
    },
    // Metadata Changes
    UpdateValidatorMetadata(UpdateValidatorMetadataArgs),
    SetCommissionRate {
        new_rate: u64,
    },

    // Coin and object transactions
    TransferCoin {
        coin: ObjectRef,
        amount: Option<u64>,
        recipient: SomaAddress,
    },
    PayCoins {
        coins: Vec<ObjectRef>,
        amounts: Option<Vec<u64>>,
        recipients: Vec<SomaAddress>,
    },
    TransferObjects {
        objects: Vec<ObjectRef>,
        recipient: SomaAddress,
    },
    // Staking txs
    AddStake {
        address: SomaAddress,
        coin_ref: ObjectRef,
        amount: Option<u64>, // Optional to allow staking entire coin
    },
    WithdrawStake {
        staked_soma: ObjectRef,
    },

    // Model transactions
    CommitModel(CommitModelArgs),
    RevealModel(RevealModelArgs),
    CommitModelUpdate(CommitModelUpdateArgs),
    RevealModelUpdate(RevealModelUpdateArgs),
    AddStakeToModel {
        model_id: ModelId,
        coin_ref: ObjectRef,
        amount: Option<u64>,
    },
    SetModelCommissionRate {
        model_id: ModelId,
        new_rate: u64,
    },
    DeactivateModel {
        model_id: ModelId,
    },
    ReportModel {
        model_id: ModelId,
    },
    UndoReportModel {
        model_id: ModelId,
    },

    // Submission transactions
    SubmitData(SubmitDataArgs),
    ClaimRewards(ClaimRewardsArgs),
    /// Report a submission as fraudulent (validators only).
    /// Reports are stored on the Target object and accumulate until 2f+1 quorum.
    /// Quorum triggers slashing at ClaimRewards time.
    ReportSubmission {
        target_id: TargetId,
        /// Optional challenger to attribute fraud to.
        /// If 2f+1 reports agree on this challenger, they get the miner's bond.
        /// If None (availability case), reporting validators split the bond.
        challenger: Option<SomaAddress>,
    },
    /// Undo a previous submission report.
    UndoReportSubmission {
        target_id: TargetId,
    },

    // Challenge transactions
    InitiateChallenge(InitiateChallengeArgs),
    /// Report that a challenge is invalid (validators only).
    /// Validators submit this when they determine the challenger's claim is wrong
    /// (i.e., the submission is actually valid). Reports accumulate on the Challenge object.
    /// If 2f+1 validators report, the challenger loses their bond.
    ReportChallenge {
        challenge_id: ChallengeId,
    },
    /// Undo a previous challenge report.
    UndoReportChallenge {
        challenge_id: ChallengeId,
    },
    /// Claim the challenger's bond after challenge window closes.
    /// Distributes the challenger's bond based on report quorum:
    /// - 2f+1 reports: challenger bond → reporting validators (challenger loses)
    /// - No quorum: challenger bond → challenger (benefit of doubt)
    ///   Note: The miner's bond is handled separately by ClaimRewards based on submission reports.
    ClaimChallengeBond {
        challenge_id: ChallengeId,
    },
}

/// # AddValidatorArgs
///
/// Contains the necessary information to add a new validator to the network.
///
/// ## Purpose
/// Encapsulates all required validator credentials and network addresses
/// needed to register a new validator in the validator set.
///
/// ## Lifecycle
/// Created as part of an AddValidator transaction, processed during transaction
/// execution, and results in a new validator being added to the validator set.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct AddValidatorArgs {
    /// The validator's public key bytes
    pub pubkey_bytes: Vec<u8>,
    /// The validator's network public key bytes for secure communication
    pub network_pubkey_bytes: Vec<u8>,
    /// The worker node's public key bytes
    pub worker_pubkey_bytes: Vec<u8>,
    /// The validator's network address
    pub net_address: Vec<u8>,
    /// The validator's peer-to-peer communication address
    pub p2p_address: Vec<u8>,
    /// The validator's primary address for client communication
    pub primary_address: Vec<u8>,
    /// The validator's proxy server address for data/model serving
    pub proxy_address: Vec<u8>,
}

/// # RemoveValidatorArgs
///
/// Contains the necessary information to remove a validator from the network.
///
/// ## Purpose
/// Identifies the validator to be removed from the validator set
/// using their public key.
///
/// ## Lifecycle
/// Created as part of a RemoveValidator transaction, processed during transaction
/// execution, and results in a validator being removed from the validator set.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct RemoveValidatorArgs {
    /// The public key bytes of the validator to remove
    pub pubkey_bytes: Vec<u8>,
}

#[derive(Debug, Default, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct UpdateValidatorMetadataArgs {
    /// Optional new network address (serialized Multiaddr)
    pub next_epoch_network_address: Option<Vec<u8>>,
    /// Optional new P2P address (serialized Multiaddr)
    pub next_epoch_p2p_address: Option<Vec<u8>>,
    /// Optional new primary address (serialized Multiaddr)
    pub next_epoch_primary_address: Option<Vec<u8>>,
    /// Optional new proxy address (serialized Multiaddr)
    pub next_epoch_proxy_address: Option<Vec<u8>>,

    /// Optional new protocol public key (BLS)
    pub next_epoch_protocol_pubkey: Option<Vec<u8>>,
    /// Optional new worker public key (Ed25519)
    pub next_epoch_worker_pubkey: Option<Vec<u8>>,
    /// Optional new network public key (Ed25519)
    pub next_epoch_network_pubkey: Option<Vec<u8>>,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct CommitModelArgs {
    pub model_id: ModelId,
    pub weights_url_commitment: ModelWeightsUrlCommitment,
    pub weights_commitment: ModelWeightsCommitment,
    pub architecture_version: ArchitectureVersion,
    pub stake_amount: u64,
    pub commission_rate: u64,
    pub staking_pool_id: ObjectID,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct RevealModelArgs {
    pub model_id: ModelId,
    pub weights_manifest: ModelWeightsManifest,
    /// Model embedding for stake-weighted KNN target selection.
    /// This positions the model in the shared embedding space.
    pub embedding: SomaTensor,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct CommitModelUpdateArgs {
    pub model_id: ModelId,
    pub weights_url_commitment: ModelWeightsUrlCommitment,
    pub weights_commitment: ModelWeightsCommitment,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct RevealModelUpdateArgs {
    pub model_id: ModelId,
    pub weights_manifest: ModelWeightsManifest,
    /// Updated model embedding for stake-weighted KNN target selection.
    /// Allows repositioning the model in the shared embedding space.
    pub embedding: SomaTensor,
}

/// Arguments for a data submission to a target.
///
/// This is a single-transaction submission (no commit-reveal) where the miner
/// provides all required data upfront. Front-running mitigation is deferred to
/// future versions.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct SubmitDataArgs {
    /// Target to submit against (shared object ID)
    pub target_id: TargetId,

    /// Commitment to the raw data: hash(data_bytes)
    pub data_commitment: DataCommitment,

    /// Manifest for submitted data (URL + checksum + size)
    pub data_manifest: SubmissionManifest,

    /// Which model the miner chose from the target's model_ids
    pub model_id: ModelId,

    /// Pre-computed embedding as SomaTensor (f32 values)
    pub embedding: SomaTensor,

    /// Distance score as SomaTensor (scalar, shape [1]). Lower is better.
    pub distance_score: SomaTensor,

    /// Coin to use for bond payment (must cover submission_bond_per_byte * data_manifest.size)
    pub bond_coin: ObjectRef,
}

/// Arguments for claiming rewards from a filled or expired target.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct ClaimRewardsArgs {
    /// Target to claim rewards from
    pub target_id: TargetId,
}

/// Arguments for initiating a fraud challenge against a filled target's submission.
///
/// The challenger must provide a bond (challenger_bond_per_byte * data_size).
/// If the challenge succeeds (fraud proven), the miner's bond is slashed and the challenger is rewarded.
/// If the challenge fails (no fraud), the challenger's bond is slashed.
///
/// **Note**: All challenges are fraud challenges. Availability issues are handled via
/// submission reports (ReportSubmission/UndoReportSubmission) instead.
///
/// The ChallengeId is derived from the transaction digest during execution (not client-provided).
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct InitiateChallengeArgs {
    /// Target being challenged (must be filled and within challenge window)
    pub target_id: TargetId,

    /// Coin to pay challenger bond (must cover challenger_bond_per_byte * data_size)
    pub bond_coin: ObjectRef,
}

impl TransactionKind {
    pub fn is_system_tx(&self) -> bool {
        matches!(
            self,
            TransactionKind::Genesis(_)
                | TransactionKind::ConsensusCommitPrologue(_)
                | TransactionKind::ChangeEpoch(_)
        )
    }

    pub fn is_validator_tx(&self) -> bool {
        matches!(
            self,
            TransactionKind::AddValidator(_)
                | TransactionKind::RemoveValidator(_)
                | TransactionKind::ReportValidator { .. }
                | TransactionKind::UndoReportValidator { .. }
                | TransactionKind::SetCommissionRate { .. }
                | TransactionKind::UpdateValidatorMetadata(_)
        )
    }

    pub fn is_staking_tx(&self) -> bool {
        matches!(self, TransactionKind::AddStake { .. } | TransactionKind::WithdrawStake { .. })
    }

    pub fn is_model_tx(&self) -> bool {
        matches!(
            self,
            TransactionKind::CommitModel(_)
                | TransactionKind::RevealModel(_)
                | TransactionKind::CommitModelUpdate(_)
                | TransactionKind::RevealModelUpdate(_)
                | TransactionKind::AddStakeToModel { .. }
                | TransactionKind::SetModelCommissionRate { .. }
                | TransactionKind::DeactivateModel { .. }
                | TransactionKind::ReportModel { .. }
                | TransactionKind::UndoReportModel { .. }
        )
    }

    pub fn is_end_of_epoch_tx(&self) -> bool {
        matches!(self, TransactionKind::ChangeEpoch(_))
    }

    pub fn is_submission_tx(&self) -> bool {
        matches!(
            self,
            TransactionKind::SubmitData(_)
                | TransactionKind::ClaimRewards(_)
                | TransactionKind::ReportSubmission { .. }
                | TransactionKind::UndoReportSubmission { .. }
        )
    }

    pub fn is_challenge_tx(&self) -> bool {
        matches!(
            self,
            TransactionKind::InitiateChallenge(_)
                | TransactionKind::ReportChallenge { .. }
                | TransactionKind::UndoReportChallenge { .. }
                | TransactionKind::ClaimChallengeBond { .. }
        )
    }

    pub fn requires_system_state(&self) -> bool {
        self.is_validator_tx()
            || self.is_epoch_change()
            || self.is_staking_tx()
            || self.is_model_tx()
            || self.is_submission_tx()
            || self.is_challenge_tx()
    }

    pub fn is_epoch_change(&self) -> bool {
        matches!(self, TransactionKind::ChangeEpoch(_))
    }

    pub fn contains_shared_object(&self) -> bool {
        self.shared_input_objects().next().is_some()
    }
    pub fn receiving_objects(&self) -> Vec<ObjectRef> {
        // Implementation for collecting receiving objects
        // For now, return empty as current transaction types don't use this
        vec![]
    }

    /// Returns an iterator of all shared input objects used by this transaction.
    pub fn shared_input_objects(&self) -> impl Iterator<Item = SharedInputObject> + '_ {
        let mut objects: Vec<SharedInputObject> = Vec::new();

        // Add system object if needed
        if self.requires_system_state() {
            objects.push(SharedInputObject::SYSTEM_OBJ);
        }

        // Add transaction-specific shared objects
        match self {
            TransactionKind::SubmitData(args) => {
                objects.push(SharedInputObject {
                    id: args.target_id,
                    initial_shared_version: crate::TARGET_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }
            TransactionKind::ClaimRewards(args) => {
                objects.push(SharedInputObject {
                    id: args.target_id,
                    initial_shared_version: crate::TARGET_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }
            TransactionKind::InitiateChallenge(args) => {
                // Challenge reads target info and updates it with challenger/challenge_id
                objects.push(SharedInputObject {
                    id: args.target_id,
                    initial_shared_version: crate::TARGET_OBJECT_SHARED_VERSION,
                    mutable: true, // writes challenger and challenge_id to target
                });
            }
            TransactionKind::ReportSubmission { target_id, .. } => {
                // ReportSubmission writes to the Target object
                objects.push(SharedInputObject {
                    id: *target_id,
                    initial_shared_version: crate::TARGET_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }
            TransactionKind::UndoReportSubmission { target_id } => {
                // UndoReportSubmission writes to the Target object
                objects.push(SharedInputObject {
                    id: *target_id,
                    initial_shared_version: crate::TARGET_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }
            TransactionKind::ReportChallenge { challenge_id, .. } => {
                // ReportChallenge writes to the Challenge object
                objects.push(SharedInputObject {
                    id: *challenge_id,
                    initial_shared_version: crate::CHALLENGE_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }
            TransactionKind::UndoReportChallenge { challenge_id } => {
                // UndoReportChallenge writes to the Challenge object
                objects.push(SharedInputObject {
                    id: *challenge_id,
                    initial_shared_version: crate::CHALLENGE_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }
            TransactionKind::ClaimChallengeBond { challenge_id } => {
                // ClaimChallengeBond writes to the Challenge object
                objects.push(SharedInputObject {
                    id: *challenge_id,
                    initial_shared_version: crate::CHALLENGE_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }
            _ => {}
        }

        objects.into_iter()
    }

    /// Return the metadata of each of the input objects for the transaction.
    /// For a Move object, we attach the object reference;
    /// for a Move package, we provide the object id only since they never change on chain.
    /// TODO: use an iterator over references here instead of a Vec to avoid allocations.
    pub fn input_objects(&self) -> SomaResult<Vec<InputObjectKind>> {
        let mut input_objects = Vec::new();

        // Add system state object if needed
        if self.requires_system_state() {
            input_objects.push(InputObjectKind::SharedObject {
                id: SYSTEM_STATE_OBJECT_ID,
                initial_shared_version: SYSTEM_STATE_OBJECT_SHARED_VERSION,
                mutable: true,
            });
        }

        // Add transaction-specific inputs
        match self {
            TransactionKind::TransferCoin { coin, .. } => {
                input_objects.push(InputObjectKind::ImmOrOwnedObject(*coin));
            }
            TransactionKind::PayCoins { coins, .. } => {
                for coin in coins {
                    input_objects.push(InputObjectKind::ImmOrOwnedObject(*coin));
                }
            }
            TransactionKind::TransferObjects { objects, .. } => {
                for object in objects {
                    input_objects.push(InputObjectKind::ImmOrOwnedObject(*object));
                }
            }
            TransactionKind::AddStake { coin_ref, .. } => {
                input_objects.push(InputObjectKind::ImmOrOwnedObject(*coin_ref));
            }

            TransactionKind::WithdrawStake { staked_soma } => {
                input_objects.push(InputObjectKind::ImmOrOwnedObject(*staked_soma));
            }

            TransactionKind::AddStakeToModel { coin_ref, .. } => {
                input_objects.push(InputObjectKind::ImmOrOwnedObject(*coin_ref));
            }

            TransactionKind::SubmitData(args) => {
                // Add bond coin as owned object
                input_objects.push(InputObjectKind::ImmOrOwnedObject(args.bond_coin));
                // Add target as shared object
                input_objects.push(InputObjectKind::SharedObject {
                    id: args.target_id,
                    initial_shared_version: crate::TARGET_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }

            TransactionKind::ClaimRewards(args) => {
                // Add target as shared object
                input_objects.push(InputObjectKind::SharedObject {
                    id: args.target_id,
                    initial_shared_version: crate::TARGET_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }

            TransactionKind::InitiateChallenge(args) => {
                // Add bond coin as owned object
                input_objects.push(InputObjectKind::ImmOrOwnedObject(args.bond_coin));
                // Add target as shared object (mutable - writes challenger and challenge_id)
                input_objects.push(InputObjectKind::SharedObject {
                    id: args.target_id,
                    initial_shared_version: crate::TARGET_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }

            TransactionKind::ReportSubmission { target_id, .. } => {
                // Add target as shared object (mutable - storing reports)
                input_objects.push(InputObjectKind::SharedObject {
                    id: *target_id,
                    initial_shared_version: crate::TARGET_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }

            TransactionKind::UndoReportSubmission { target_id } => {
                // Add target as shared object (mutable - removing reports)
                input_objects.push(InputObjectKind::SharedObject {
                    id: *target_id,
                    initial_shared_version: crate::TARGET_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }

            TransactionKind::ReportChallenge { challenge_id, .. } => {
                // Add challenge as shared object (mutable - storing reports)
                input_objects.push(InputObjectKind::SharedObject {
                    id: *challenge_id,
                    initial_shared_version: crate::CHALLENGE_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }

            TransactionKind::UndoReportChallenge { challenge_id } => {
                // Add challenge as shared object (mutable - removing reports)
                input_objects.push(InputObjectKind::SharedObject {
                    id: *challenge_id,
                    initial_shared_version: crate::CHALLENGE_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }

            TransactionKind::ClaimChallengeBond { challenge_id } => {
                // Add challenge as shared object (mutable - resolving and clearing reports)
                input_objects.push(InputObjectKind::SharedObject {
                    id: *challenge_id,
                    initial_shared_version: crate::CHALLENGE_OBJECT_SHARED_VERSION,
                    mutable: true,
                });
            }

            _ => {}
        }

        // Ensure that there are no duplicate inputs. This cannot be removed because:
        // In [`AuthorityState::check_locks`], we check that there are no duplicate mutable
        // input objects, which would have made this check here unnecessary. However we
        // do plan to allow shared objects show up more than once in multiple single
        // transactions down the line. Once we have that, we need check here to make sure
        // the same shared object doesn't show up more than once in the same single
        // transaction.
        let mut used = HashSet::new();
        if !input_objects.iter().all(|o| used.insert(o.object_id())) {
            return Err(SomaError::DuplicateObjectRefInput);
        }
        Ok(input_objects)
    }
}

/// # GenesisTransaction
///
/// Represents the initial transaction that creates the genesis state of the blockchain.
///
/// ## Purpose
/// Defines the initial set of objects that exist at blockchain creation,
/// establishing the foundation state from which all future transactions build.
///
/// ## Lifecycle
/// Created once at blockchain initialization and executed as the first transaction
/// in the blockchain's history.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct GenesisTransaction {
    /// The initial set of objects to be created in the genesis state
    pub objects: Vec<Object>,
}

/// # ChangeEpoch
///
/// Contains the information needed to transition the blockchain to a new epoch.
///
/// ## Purpose
/// Defines the parameters for an epoch change, including the new epoch ID
/// and the timestamp when the epoch started.
///
/// ## Lifecycle
/// Created at the end of an epoch, processed as part of an EndOfEpochTransaction,
/// and results in the blockchain transitioning to a new epoch.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct ChangeEpoch {
    /// The next (to become) epoch ID.
    pub epoch: EpochId,
    /// Unix timestamp when epoch started
    pub epoch_start_timestamp_ms: u64,
    /// The protocol version in effect in the new epoch.
    pub protocol_version: ProtocolVersion,
    /// The total amount of fees charged during the epoch.
    pub fees: u64,
    /// Epoch randomness
    pub epoch_randomness: Vec<u8>,
}

/// # CertificateProof
///
/// A proof that a transaction certificate existed at a given epoch and hence can be executed.
///
/// ## Purpose
/// Provides cryptographic evidence that a transaction has been properly certified
/// and is valid for execution, using one of several proof mechanisms.
///
/// ## Variants
/// - `Commit`: Validity proven by inclusion in a specific commit
/// - `Certified`: Validity proven by transaction certificate signature
/// - `QuorumExecuted`: Validity proven by execution by a quorum of validators
/// - `SystemTransaction`: System-generated transaction that doesn't require external validation
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CertificateProof {
    /// Validity was proven by inclusion in the given checkpoint
    Checkpoint(EpochId, CheckpointSequenceNumber),
    /// Validity was proven by transaction certificate signature
    Certified(AuthorityStrongQuorumSignInfo),
    /// At least f+1 validators have executed this transaction.
    /// In practice, we will always get 2f+1 (effects cert), but theoretically f+1 is enough to prove
    /// that the transaction is valid.
    QuorumExecuted(EpochId),
    /// Transaction generated by the system, for example Clock update transaction
    SystemTransaction(EpochId),
    /// Validity was proven through voting in consensus.
    Consensus(EpochId),
}

impl CertificateProof {
    pub fn new_from_cert_sig(sig: AuthorityStrongQuorumSignInfo) -> Self {
        Self::Certified(sig)
    }

    pub fn new_from_checkpoint(epoch: EpochId, checkpoint: CheckpointSequenceNumber) -> Self {
        Self::Checkpoint(epoch, checkpoint)
    }

    pub fn new_system(epoch: EpochId) -> Self {
        Self::SystemTransaction(epoch)
    }

    pub fn new_from_consensus(epoch: EpochId) -> Self {
        Self::Consensus(epoch)
    }

    pub fn epoch(&self) -> EpochId {
        match self {
            Self::Checkpoint(epoch, _)
            | Self::QuorumExecuted(epoch)
            | Self::SystemTransaction(epoch)
            | Self::Consensus(epoch) => *epoch,
            Self::Certified(sig) => sig.epoch,
        }
    }
}

/// # ExecutableTransaction
///
/// A wrapper of a transaction with a CertificateProof that indicates
/// there existed a valid certificate for this transaction, and hence it can be executed locally.
///
/// ## Purpose
/// Provides an abstraction for transactions that are ready for execution,
/// whether they were certified or checkpointed when scheduled for execution.
///
/// ## Related Types
/// - `VerifiedExecutableTransaction`: An ExecutableTransaction that has been verified
/// - `TrustedExecutableTransaction`: An ExecutableTransaction that is trusted without verification
pub type ExecutableTransaction = Envelope<SenderSignedData, CertificateProof>;
pub type VerifiedExecutableTransaction = VerifiedEnvelope<SenderSignedData, CertificateProof>;
pub type TrustedExecutableTransaction = TrustedEnvelope<SenderSignedData, CertificateProof>;

/// # Transaction
///
/// A transaction that is signed by a sender but not yet by an authority.
///
/// ## Purpose
/// Represents the initial state of a transaction after it has been created and
/// signed by a user but before it has been processed by the network.
///
/// ## Related Types
/// - `VerifiedTransaction`: A Transaction that has been verified
/// - `TrustedTransaction`: A Transaction that is trusted without verification
pub type Transaction = Envelope<SenderSignedData, EmptySignInfo>;
pub type VerifiedTransaction = VerifiedEnvelope<SenderSignedData, EmptySignInfo>;
pub type TrustedTransaction = TrustedEnvelope<SenderSignedData, EmptySignInfo>;

impl Transaction {
    pub fn verify_signature_for_testing(&self) -> SomaResult {
        verify_sender_signed_data_message_signatures(self.data())
    }

    pub fn try_into_verified_for_testing(self) -> SomaResult<VerifiedTransaction> {
        Ok(VerifiedTransaction::new_from_verified(self))
    }
}

/// # SignedTransaction
///
/// A transaction that is signed by a sender and also by an authority.
///
/// ## Purpose
/// Represents a transaction that has been processed by a single authority
/// but has not yet received enough signatures to form a certificate.
///
/// ## Related Types
/// - `VerifiedSignedTransaction`: A SignedTransaction that has been verified
pub type SignedTransaction = Envelope<SenderSignedData, AuthoritySignInfo>;
pub type VerifiedSignedTransaction = VerifiedEnvelope<SenderSignedData, AuthoritySignInfo>;

impl Transaction {
    pub fn from_data_and_signer(
        data: TransactionData,
        signers: Vec<&dyn Signer<Signature>>,
    ) -> Self {
        let signatures = {
            let intent_msg = IntentMessage::new(Intent::soma_transaction(), &data);
            signers.into_iter().map(|s| Signature::new_secure(&intent_msg, s)).collect()
        };
        Self::from_data(data, signatures)
    }

    pub fn from_data(data: TransactionData, signatures: Vec<Signature>) -> Self {
        Self::from_generic_sig_data(data, signatures.into_iter().map(|s| s.into()).collect())
    }

    pub fn from_generic_sig_data(data: TransactionData, signatures: Vec<GenericSignature>) -> Self {
        Self::new(SenderSignedData::new(data, signatures))
    }
}

impl SenderSignedData {
    pub fn new(tx_data: TransactionData, tx_signatures: Vec<GenericSignature>) -> Self {
        Self(SizeOneVec::new(SenderSignedTransaction {
            intent_message: IntentMessage::new(Intent::soma_transaction(), tx_data),
            tx_signatures,
        }))
    }
}

impl VerifiedSignedTransaction {
    /// Use signing key to create a signed object.
    pub fn new(
        epoch: EpochId,
        transaction: VerifiedTransaction,
        authority: AuthorityName,
        secret: &dyn Signer<AuthoritySignature>,
    ) -> Self {
        Self::new_from_verified(SignedTransaction::new(
            epoch,
            transaction.into_inner().into_data(),
            secret,
            authority,
        ))
    }
}

/// # TransactionData
///
/// Contains the core data of a transaction, including its type and sender.
///
/// ## Purpose
/// Encapsulates the essential information needed to define a transaction,
/// separate from signatures and other metadata.
///
/// ## Lifecycle
/// Created during transaction construction, signed by the sender,
/// and eventually executed by the network.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct TransactionData {
    /// The specific type of transaction
    pub kind: TransactionKind,
    /// The address of the transaction sender
    pub sender: SomaAddress,

    pub gas_payment: Vec<ObjectRef>,
}

impl TransactionData {
    pub fn new(kind: TransactionKind, sender: SomaAddress, gas_payment: Vec<ObjectRef>) -> Self {
        TransactionData { kind, sender, gas_payment }
    }

    fn new_system_transaction(kind: TransactionKind) -> Self {
        // assert transaction kind if a system transaction
        assert!(kind.is_system_tx());
        let sender = SomaAddress::default();
        TransactionData { kind, sender, gas_payment: vec![] }
    }

    pub fn new_pay_coins(
        coins: Vec<ObjectRef>,
        amounts: Option<Vec<u64>>,
        recipients: Vec<SomaAddress>,
        sender: SomaAddress,
    ) -> Self {
        // Use the first coin in the list as gas payment
        if coins.is_empty() {
            panic!("PayCoins transaction must have at least one coin");
        }
        Self::new(
            TransactionKind::PayCoins { coins: coins.clone(), amounts, recipients },
            sender,
            vec![coins[0]],
        )
    }

    pub fn new_transfer_coin(
        recipient: SomaAddress,
        sender: SomaAddress,
        amount: Option<u64>,
        object_ref: ObjectRef,
    ) -> Self {
        Self::new(
            TransactionKind::TransferCoin { coin: object_ref, amount, recipient },
            sender,
            vec![object_ref],
        )
    }

    pub fn new_transfer(
        recipient: SomaAddress,
        object_ref: ObjectRef,
        sender: SomaAddress,
        gas_payment: Vec<ObjectRef>,
    ) -> Self {
        Self::new(
            TransactionKind::TransferObjects { objects: vec![object_ref], recipient },
            sender,
            gas_payment,
        )
    }

    pub fn digest(&self) -> TransactionDigest {
        TransactionDigest::new(default_hash(self))
    }

    pub fn signers(&self) -> NonEmpty<SomaAddress> {
        let mut signers = nonempty![self.sender];
        // if self.gas_owner() != self.sender {
        //     signers.push(self.gas_owner());
        // }
        signers
    }

    pub fn is_system_tx(&self) -> bool {
        self.kind.is_system_tx()
    }

    pub fn is_genesis_tx(&self) -> bool {
        matches!(self.kind, TransactionKind::Genesis(_))
    }

    pub fn is_consensus_commit_prologue(&self) -> bool {
        matches!(&self.kind, TransactionKind::ConsensusCommitPrologue(_))
    }

    pub fn execution_parts(&self) -> (TransactionKind, SomaAddress, Vec<ObjectRef>) {
        (self.kind().clone(), self.sender(), self.gas())
    }

    pub fn kind(&self) -> &TransactionKind {
        &self.kind
    }

    pub fn sender(&self) -> SomaAddress {
        self.sender
    }

    pub fn gas(&self) -> Vec<ObjectRef> {
        self.gas_payment.clone()
    }

    pub fn gas_mut(&mut self) -> &mut Vec<ObjectRef> {
        &mut self.gas_payment
    }

    fn contains_shared_object(&self) -> bool {
        self.kind.shared_input_objects().next().is_some()
    }

    pub fn shared_input_objects(&self) -> Vec<SharedInputObject> {
        self.kind.shared_input_objects().collect()
    }

    pub fn input_objects(&self) -> SomaResult<Vec<InputObjectKind>> {
        // Get inputs from transaction kind
        let mut inputs = self.kind.input_objects()?;

        // For non-system transactions, add gas objects not already included in inputs
        if !self.kind.is_system_tx() {
            // Create a set of object IDs already in the inputs
            let input_object_ids: HashSet<ObjectID> =
                inputs.iter().map(|input| input.object_id()).collect();

            // Only add gas objects that aren't already in the inputs
            for gas_ref in &self.gas_payment {
                if !input_object_ids.contains(&gas_ref.0) {
                    inputs.push(InputObjectKind::ImmOrOwnedObject(*gas_ref));
                }
            }
        }

        // Check for duplicates in the combined list
        let mut used = HashSet::new();
        if !inputs.iter().all(|o| used.insert(o.object_id())) {
            return Err(SomaError::DuplicateObjectRefInput);
        }

        Ok(inputs)
    }

    pub fn receiving_objects(&self) -> Vec<ObjectRef> {
        self.kind.receiving_objects()
    }

    // Dependency (input, package & receiving) objects that already have a version,
    // and do not require version assignment from consensus.
    // Returns objects and receiving objects.
    pub fn fastpath_dependency_objects(&self) -> SomaResult<(Vec<ObjectRef>, Vec<ObjectRef>)> {
        let mut objects = vec![];

        let mut receiving_objects = vec![];
        self.input_objects()?.iter().for_each(|o| match o {
            InputObjectKind::ImmOrOwnedObject(object_ref) => {
                objects.push(*object_ref);
            }
            InputObjectKind::SharedObject { .. } => {}
        });
        self.receiving_objects().iter().for_each(|object_ref| {
            receiving_objects.push(*object_ref);
        });
        Ok((objects, receiving_objects))
    }
}

/// # SenderSignedData
///
/// Contains transaction data signed by the sender.
///
/// ## Purpose
/// Wraps the transaction data and signatures from the sender,
/// providing a container for the core transaction information.
///
/// ## Lifecycle
/// Created when a transaction is signed by the sender,
/// and used throughout the transaction's lifecycle.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SenderSignedData(SizeOneVec<SenderSignedTransaction>);

impl SenderSignedData {
    pub fn inner(&self) -> &SenderSignedTransaction {
        self.0.element()
    }

    pub fn into_inner(self) -> SenderSignedTransaction {
        self.0.into_inner()
    }

    pub fn intent_message(&self) -> &IntentMessage<TransactionData> {
        self.inner().intent_message()
    }

    pub fn new_from_sender_signature(tx_data: TransactionData, tx_signature: Signature) -> Self {
        Self(SizeOneVec::new(SenderSignedTransaction {
            intent_message: IntentMessage::new(Intent::soma_transaction(), tx_data),
            tx_signatures: vec![tx_signature.into()],
        }))
    }

    pub fn transaction_data(&self) -> &TransactionData {
        &self.intent_message().value
    }

    pub(crate) fn get_signer_sig_mapping(
        &self,
    ) -> SomaResult<BTreeMap<SomaAddress, &GenericSignature>> {
        self.inner().get_signer_sig_mapping()
    }

    pub fn tx_signatures(&self) -> &[GenericSignature] {
        &self.inner().tx_signatures
    }

    pub fn full_message_digest(&self) -> SenderSignedDataDigest {
        let mut digest = DefaultHash::default();
        bcs::serialize_into(&mut digest, self).expect("serialization should not fail");
        let hash = digest.finalize();
        SenderSignedDataDigest::new(hash.into())
    }

    pub fn serialized_size(&self) -> SomaResult<usize> {
        bcs::serialized_size(self)
            .map_err(|e| SomaError::TransactionSerializationError { error: e.to_string() })
    }
}

impl Message for SenderSignedData {
    type DigestType = TransactionDigest;
    const SCOPE: IntentScope = IntentScope::SenderSignedTransaction;

    /// Computes the tx digest that encodes the Rust type prefix from Signable trait.
    fn digest(&self) -> Self::DigestType {
        TransactionDigest::new(default_hash(&self.intent_message().value))
    }
}

impl<S> Envelope<SenderSignedData, S> {
    // Returns the primary key for this transaction.
    pub fn key(&self) -> TransactionKey {
        TransactionKey::Digest(*self.digest())
    }

    pub fn contains_shared_object(&self) -> bool {
        !self.data().inner().intent_message.value.shared_input_objects().is_empty()
    }

    pub fn shared_input_objects(&self) -> impl Iterator<Item = SharedInputObject> + '_ {
        self.data().inner().intent_message.value.shared_input_objects().into_iter()
    }

    pub fn is_consensus_tx(&self) -> bool {
        self.shared_input_objects().next().is_some()
    }
}

/// # SenderSignedTransaction
///
/// Represents a transaction that has been signed by the sender.
///
/// ## Purpose
/// Combines the transaction data (wrapped in an intent message) with
/// the signatures from all transaction participants.
///
/// ## Lifecycle
/// Created when a transaction is signed by the sender,
/// and used as the basis for further processing by authorities.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SenderSignedTransaction {
    /// The transaction data wrapped in an intent message
    pub intent_message: IntentMessage<TransactionData>,
    /// A list of signatures signed by all transaction participants.
    /// 1. non participant signature must not be present.
    /// 2. signature order does not matter.
    pub tx_signatures: Vec<GenericSignature>,
}

impl Serialize for SenderSignedTransaction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        #[serde(rename = "SenderSignedTransaction")]
        struct SignedTxn<'a> {
            intent_message: &'a IntentMessage<TransactionData>,
            tx_signatures: &'a Vec<GenericSignature>,
        }

        // TODO: if self.intent_message().intent != Intent::transaction() {
        //     return Err(serde::ser::Error::custom("invalid Intent for Transaction"));
        // }

        let txn =
            SignedTxn { intent_message: self.intent_message(), tx_signatures: &self.tx_signatures };
        txn.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SenderSignedTransaction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename = "SenderSignedTransaction")]
        struct SignedTxn {
            intent_message: IntentMessage<TransactionData>,
            tx_signatures: Vec<GenericSignature>,
        }

        let SignedTxn { intent_message, tx_signatures } = Deserialize::deserialize(deserializer)?;

        // TODO: if intent_message.intent != Intent::transaction() {
        //     return Err(serde::de::Error::custom("invalid Intent for Transaction"));
        // }

        Ok(Self { intent_message, tx_signatures })
    }
}

impl SenderSignedTransaction {
    pub fn intent_message(&self) -> &IntentMessage<TransactionData> {
        &self.intent_message
    }

    pub(crate) fn get_signer_sig_mapping(
        &self,
    ) -> SomaResult<BTreeMap<SomaAddress, &GenericSignature>> {
        let mut mapping = BTreeMap::new();
        for sig in &self.tx_signatures {
            let address = sig.try_into()?;
            mapping.insert(address, sig);
        }
        Ok(mapping)
    }
}

/// # CertifiedTransaction
///
/// A transaction that has been certified by a quorum of validators.
///
/// ## Purpose
/// Represents a transaction that has received enough signatures from validators
/// to form a certificate, making it ready for execution.
///
/// ## Lifecycle
/// Created when a transaction receives signatures from a quorum of validators,
/// and used as the basis for transaction execution.
pub type CertifiedTransaction = Envelope<SenderSignedData, AuthorityStrongQuorumSignInfo>;

impl CertifiedTransaction {
    pub fn certificate_digest(&self) -> CertificateDigest {
        let mut digest = DefaultHash::default();
        bcs::serialize_into(&mut digest, self).expect("serialization should not fail");
        let hash = digest.finalize();
        CertificateDigest::new(hash.into())
    }

    pub fn verify_signatures_authenticated(&self, committee: &Committee) -> SomaResult {
        verify_sender_signed_data_message_signatures(self.data())?;
        self.auth_sig().verify_secure(self.data(), Intent::soma_transaction(), committee)
    }

    pub fn verify_committee_sigs_only(&self, committee: &Committee) -> SomaResult {
        self.auth_sig().verify_secure(self.data(), Intent::soma_transaction(), committee)
    }

    pub fn try_into_verified_for_testing(
        self,
        committee: &Committee,
    ) -> SomaResult<VerifiedCertificate> {
        self.verify_signatures_authenticated(committee)?;
        Ok(VerifiedCertificate::new_from_verified(self))
    }
}

impl VerifiedTransaction {
    pub fn new_genesis_transaction(objects: Vec<Object>) -> Self {
        GenesisTransaction { objects }
            .pipe(TransactionKind::Genesis)
            .pipe(Self::new_system_transaction)
    }

    pub fn new_consensus_commit_prologue(
        epoch: u64,
        round: u64,
        commit_timestamp_ms: CheckpointTimestamp,
        consensus_commit_digest: ConsensusCommitDigest,
        additional_state_digest: AdditionalConsensusStateDigest,
    ) -> Self {
        ConsensusCommitPrologue {
            epoch,
            round,
            // sub_dag_index is reserved for when we have multi commits per round.
            sub_dag_index: None,
            commit_timestamp_ms,
            consensus_commit_digest,
            additional_state_digest,
        }
        .pipe(TransactionKind::ConsensusCommitPrologue)
        .pipe(Self::new_system_transaction)
    }

    pub fn new_change_epoch_transaction(
        next_epoch: EpochId,
        protocol_version: ProtocolVersion,
        fees: u64,
        epoch_start_timestamp_ms: u64,
        epoch_randomness: Vec<u8>,
    ) -> Self {
        TransactionKind::ChangeEpoch(ChangeEpoch {
            epoch: next_epoch,
            protocol_version,
            fees,
            epoch_start_timestamp_ms,
            epoch_randomness,
        })
        .pipe(Self::new_system_transaction)
    }

    pub fn new_system_transaction(system_transaction: TransactionKind) -> Self {
        system_transaction
            .pipe(TransactionData::new_system_transaction)
            .pipe(|data| {
                SenderSignedData::new_from_sender_signature(
                    data,
                    Ed25519SomaSignature::from_bytes(&[0; Ed25519SomaSignature::LENGTH])
                        .unwrap()
                        .into(),
                )
            })
            .pipe(Transaction::new)
            .pipe(Self::new_from_verified)
    }
}

pub type VerifiedCertificate = VerifiedEnvelope<SenderSignedData, AuthorityStrongQuorumSignInfo>;

/// Does crypto validation for a transaction which may be user-provided, or may be from a checkpoint.
pub fn verify_sender_signed_data_message_signatures(txn: &SenderSignedData) -> SomaResult {
    let intent_message = txn.intent_message();
    assert_eq!(intent_message.intent, Intent::soma_transaction());

    // 1. System transactions do not require signatures. User-submitted transactions are verified not to
    // be system transactions before this point
    if intent_message.value.is_system_tx() {
        return Ok(());
    }

    // 2. One signature per signer is required.
    let signers: NonEmpty<_> = txn.intent_message().value.signers();
    if txn.inner().tx_signatures.len() != signers.len() {
        return Err(SomaError::SignerSignatureNumberMismatch {
            expected: signers.len(),
            actual: txn.inner().tx_signatures.len(),
        });
    }

    // 3. Each signer must provide a signature.
    let present_sigs = txn.get_signer_sig_mapping()?;
    for s in signers {
        if !present_sigs.contains_key(&s) {
            return Err(SomaError::SignerSignatureAbsent {
                expected: s.to_string(),
                actual: present_sigs.keys().map(|s| s.to_string()).collect(),
            });
        }
    }

    // 4. Every signature must be valid.
    for (signer, signature) in present_sigs {
        signature.verify_authenticator(intent_message, signer)?;
    }
    Ok(())
}

/// # TransactionKey
///
/// Uniquely identifies a transaction across all epochs.
///
/// ## Purpose
/// Provides a way to reference transactions uniquely throughout the system,
/// enabling transaction lookup and tracking.
///
/// ## Variants
/// - `Digest`: Identifies a transaction by its digest
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TransactionKey {
    /// Identifies a transaction by its digest
    Digest(TransactionDigest),
}

impl TransactionKey {
    pub fn unwrap_digest(&self) -> &TransactionDigest {
        match self {
            TransactionKey::Digest(d) => d,
            _ => panic!("called expect_digest on a non-Digest TransactionKey: {self:?}"),
        }
    }

    pub fn as_digest(&self) -> Option<&TransactionDigest> {
        match self {
            TransactionKey::Digest(d) => Some(d),
            _ => None,
        }
    }
}

/// # SharedInputObject
///
/// Represents a shared object that is used as input to a transaction.
///
/// ## Purpose
/// Encapsulates the information needed to reference a shared object,
/// including its ID, initial version, and mutability.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Debug, PartialEq, Eq)]
pub struct SharedInputObject {
    /// The unique identifier of the shared object
    pub id: ObjectID,
    /// The initial version when the object became shared
    pub initial_shared_version: Version,
    /// Whether the transaction requires mutable access to the object
    pub mutable: bool,
}

impl SharedInputObject {
    pub const SYSTEM_OBJ: Self = Self {
        id: SYSTEM_STATE_OBJECT_ID,
        initial_shared_version: SYSTEM_STATE_OBJECT_SHARED_VERSION,
        mutable: true,
    };

    pub fn id(&self) -> ObjectID {
        self.id
    }

    pub fn id_and_version(&self) -> (ObjectID, Version) {
        (self.id, self.initial_shared_version)
    }

    pub fn into_id_and_version(self) -> (ObjectID, Version) {
        (self.id, self.initial_shared_version)
    }
}

/// # InputObjectKind
///
/// Represents the different kinds of objects that can be used as inputs to a transaction.
///
/// ## Purpose
/// Distinguishes between different object access patterns in transactions,
/// including immutable/owned objects and shared objects.
///
/// ## Variants
/// - `ImmOrOwnedObject`: An immutable or owned object reference
/// - `SharedObject`: A shared object that may be accessed by multiple transactions
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, PartialOrd, Ord, Hash)]
pub enum InputObjectKind {
    /// A Move object, either immutable, or owned mutable
    ImmOrOwnedObject(ObjectRef),
    /// A Move object that's shared and mutable
    SharedObject {
        /// The object's unique identifier
        id: ObjectID,
        /// The initial version when the object became shared
        initial_shared_version: Version,
        /// Whether the transaction requires mutable access to the object
        mutable: bool,
    },
}

impl InputObjectKind {
    pub fn object_id(&self) -> ObjectID {
        match self {
            Self::ImmOrOwnedObject((id, _, _)) => *id,
            Self::SharedObject { id, .. } => *id,
        }
    }

    pub fn version(&self) -> Option<Version> {
        match self {
            Self::ImmOrOwnedObject((_, version, _)) => Some(*version),
            Self::SharedObject { .. } => None,
        }
    }

    pub fn object_not_found_error(&self) -> SomaError {
        match *self {
            Self::ImmOrOwnedObject((object_id, version, _)) => {
                SomaError::ObjectNotFound { object_id, version: Some(version) }
            }
            Self::SharedObject { id, .. } => {
                SomaError::ObjectNotFound { object_id: id, version: None }
            }
        }
    }

    pub fn is_shared_object(&self) -> bool {
        matches!(self, Self::SharedObject { .. })
    }

    pub fn is_mutable(&self) -> bool {
        match self {
            Self::ImmOrOwnedObject((_, _, _)) => true,
            Self::SharedObject { mutable, .. } => *mutable,
        }
    }

    pub fn full_object_id(&self) -> FullObjectID {
        match self {
            Self::ImmOrOwnedObject((id, _, _)) => FullObjectID::Fastpath(*id),
            Self::SharedObject { id, initial_shared_version, .. } => {
                FullObjectID::Consensus((*id, *initial_shared_version))
            }
        }
    }
}

/// # ObjectReadResult
///
/// The result of reading an object for execution.
///
/// ## Purpose
/// Encapsulates both the input object kind and the actual object data (or information
/// about why the object couldn't be read), providing a complete view of an object
/// for transaction execution.
///
/// ## Lifecycle
/// Created during transaction input resolution, used during transaction execution,
/// and helps determine how objects should be processed.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Clone, Debug)]
pub struct ObjectReadResult {
    /// The kind of input object (immutable/owned or shared)
    pub input_object_kind: InputObjectKind,
    /// The actual object data or information about why it couldn't be read
    pub object: ObjectReadResultKind,
}

/// # ObjectReadResultKind
///
/// Represents the different possible results of reading an object.
///
/// ## Purpose
/// Handles the various states an object might be in when a transaction attempts to read it,
/// including normal objects, deleted shared objects, and objects in cancelled transactions.
///
/// ## Variants
/// - `Object`: A normal object that exists and can be read
/// - `DeletedSharedObject`: A shared object that has been deleted
/// - `CancelledTransactionSharedObject`: A shared object in a cancelled transaction
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Clone)]
pub enum ObjectReadResultKind {
    /// A normal object that exists and can be read
    Object(Object),
    /// The version of the object that the transaction intended to read, and the digest of the tx
    /// that deleted it
    DeletedSharedObject(Version, TransactionDigest),
    /// A shared object in a cancelled transaction. The sequence number embeds cancellation reason
    CancelledTransactionSharedObject(Version),
}

impl std::fmt::Debug for ObjectReadResultKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectReadResultKind::Object(obj) => {
                write!(f, "Object({:?})", obj.compute_object_reference())
            }
            ObjectReadResultKind::DeletedSharedObject(seq, digest) => {
                write!(f, "DeletedSharedObject({}, {:?})", seq.value(), digest)
            }
            ObjectReadResultKind::CancelledTransactionSharedObject(seq) => {
                write!(f, "CancelledTransactionSharedObject({})", seq.value())
            }
        }
    }
}

impl From<Object> for ObjectReadResultKind {
    fn from(object: Object) -> Self {
        Self::Object(object)
    }
}

impl ObjectReadResult {
    pub fn new(input_object_kind: InputObjectKind, object: ObjectReadResultKind) -> Self {
        if let (
            InputObjectKind::ImmOrOwnedObject(_),
            ObjectReadResultKind::DeletedSharedObject(_, _),
        ) = (&input_object_kind, &object)
        {
            panic!("only shared objects can be DeletedSharedObject");
        }

        if let (
            InputObjectKind::ImmOrOwnedObject(_),
            ObjectReadResultKind::CancelledTransactionSharedObject(_),
        ) = (&input_object_kind, &object)
        {
            panic!("only shared objects can be CancelledTransactionSharedObject");
        }

        Self { input_object_kind, object }
    }

    pub fn id(&self) -> ObjectID {
        self.input_object_kind.object_id()
    }

    pub fn as_object(&self) -> Option<&Object> {
        match &self.object {
            ObjectReadResultKind::Object(object) => Some(object),
            ObjectReadResultKind::DeletedSharedObject(_, _) => None,
            ObjectReadResultKind::CancelledTransactionSharedObject(_) => None,
        }
    }

    pub fn new_from_gas_object(gas: &Object) -> Self {
        let objref = gas.compute_object_reference();
        Self {
            input_object_kind: InputObjectKind::ImmOrOwnedObject(objref),
            object: ObjectReadResultKind::Object(gas.clone()),
        }
    }

    pub fn is_mutable(&self) -> bool {
        match (&self.input_object_kind, &self.object) {
            (InputObjectKind::ImmOrOwnedObject(_), ObjectReadResultKind::Object(object)) => {
                !object.is_immutable()
            }
            (
                InputObjectKind::ImmOrOwnedObject(_),
                ObjectReadResultKind::DeletedSharedObject(_, _),
            ) => unreachable!(),
            (
                InputObjectKind::ImmOrOwnedObject(_),
                ObjectReadResultKind::CancelledTransactionSharedObject(_),
            ) => unreachable!(),
            (InputObjectKind::SharedObject { mutable, .. }, _) => *mutable,
        }
    }

    pub fn is_shared_object(&self) -> bool {
        self.input_object_kind.is_shared_object()
    }

    pub fn is_deleted_shared_object(&self) -> bool {
        self.deletion_info().is_some()
    }

    pub fn deletion_info(&self) -> Option<(Version, TransactionDigest)> {
        match &self.object {
            ObjectReadResultKind::DeletedSharedObject(v, tx) => Some((*v, *tx)),
            _ => None,
        }
    }

    /// Return the object ref iff the object is an owned object (i.e. not shared, not immutable).
    pub fn get_owned_objref(&self) -> Option<ObjectRef> {
        match (&self.input_object_kind, &self.object) {
            (InputObjectKind::ImmOrOwnedObject(objref), ObjectReadResultKind::Object(object)) => {
                if object.is_immutable() {
                    None
                } else {
                    Some(*objref)
                }
            }
            (
                InputObjectKind::ImmOrOwnedObject(_),
                ObjectReadResultKind::DeletedSharedObject(_, _),
            ) => unreachable!(),
            (
                InputObjectKind::ImmOrOwnedObject(_),
                ObjectReadResultKind::CancelledTransactionSharedObject(_),
            ) => unreachable!(),
            (InputObjectKind::SharedObject { .. }, _) => None,
        }
    }

    pub fn is_owned(&self) -> bool {
        self.get_owned_objref().is_some()
    }

    pub fn to_shared_input(&self) -> Option<SharedInput> {
        match self.input_object_kind {
            InputObjectKind::ImmOrOwnedObject(_) => None,
            InputObjectKind::SharedObject { id, mutable, .. } => Some(match &self.object {
                ObjectReadResultKind::Object(obj) => {
                    SharedInput::Existing(obj.compute_object_reference())
                }
                ObjectReadResultKind::DeletedSharedObject(seq, digest) => {
                    SharedInput::Deleted((id, *seq, mutable, *digest))
                }
                ObjectReadResultKind::CancelledTransactionSharedObject(seq) => {
                    SharedInput::Cancelled((id, *seq))
                }
            }),
        }
    }

    pub fn get_previous_transaction(&self) -> Option<TransactionDigest> {
        match &self.object {
            ObjectReadResultKind::Object(obj) => Some(obj.previous_transaction),
            ObjectReadResultKind::DeletedSharedObject(_, digest) => Some(*digest),
            ObjectReadResultKind::CancelledTransactionSharedObject(_) => None,
        }
    }
}

#[derive(Clone)]
pub struct InputObjects {
    objects: Vec<ObjectReadResult>,
}

impl std::fmt::Debug for InputObjects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.objects.iter()).finish()
    }
}

// An InputObjects new-type that has been verified by soma-transaction-checks, and can be
// safely passed to execution.
pub struct CheckedInputObjects(InputObjects);

// DO NOT CALL outside of soma-transaction-checks, genesis, or replay.
//
// CheckedInputObjects should really be defined in soma-transaction-checks so that we can
// make public construction impossible. But we can't do that because it would result in circular
// dependencies.
impl CheckedInputObjects {
    // Only called by soma-transaction-checks.
    pub fn new_with_checked_transaction_inputs(inputs: InputObjects) -> Self {
        Self(inputs)
    }

    // Only called when building the genesis transaction
    pub fn new_for_genesis(input_objects: Vec<ObjectReadResult>) -> Self {
        Self(InputObjects::new(input_objects))
    }

    // Only called from the replay tool.
    pub fn new_for_replay(input_objects: InputObjects) -> Self {
        Self(input_objects)
    }

    pub fn inner(&self) -> &InputObjects {
        &self.0
    }

    pub fn into_inner(self) -> InputObjects {
        self.0
    }
}

impl From<Vec<ObjectReadResult>> for InputObjects {
    fn from(objects: Vec<ObjectReadResult>) -> Self {
        Self::new(objects)
    }
}

impl InputObjects {
    pub fn new(objects: Vec<ObjectReadResult>) -> Self {
        Self { objects }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn contains_deleted_objects(&self) -> bool {
        self.objects.iter().any(|obj| obj.is_deleted_shared_object())
    }

    // Returns IDs of objects responsible for a trasnaction being cancelled, and the corresponding
    // reason for cancellation.
    pub fn get_cancelled_objects(&self) -> Option<(Vec<ObjectID>, Version)> {
        let mut contains_cancelled = false;
        let mut cancel_reason = None;
        let mut cancelled_objects = Vec::new();
        for obj in &self.objects {
            if let ObjectReadResultKind::CancelledTransactionSharedObject(version) = obj.object {
                contains_cancelled = true;
                if version == Version::CONGESTED {
                    // Verify we don't have multiple cancellation reasons.
                    assert!(cancel_reason.is_none() || cancel_reason == Some(version));
                    cancel_reason = Some(version);
                    cancelled_objects.push(obj.id());
                }
            }
        }

        if !cancelled_objects.is_empty() {
            Some((
                cancelled_objects,
                cancel_reason
                    .expect("there should be a cancel reason if there are cancelled objects"),
            ))
        } else {
            assert!(!contains_cancelled);
            None
        }
    }

    pub fn filter_owned_objects(&self) -> Vec<ObjectRef> {
        let owned_objects: Vec<_> =
            self.objects.iter().filter_map(|obj| obj.get_owned_objref()).collect();

        trace!(
            num_mutable_objects = owned_objects.len(),
            "Checked locks and found mutable objects"
        );

        owned_objects
    }

    pub fn filter_shared_objects(&self) -> Vec<SharedInput> {
        self.objects
            .iter()
            .filter(|obj| obj.is_shared_object())
            .map(|obj| obj.to_shared_input().expect("already filtered for shared objects"))
            .collect()
    }

    pub fn transaction_dependencies(&self) -> BTreeSet<TransactionDigest> {
        self.objects.iter().filter_map(|obj| obj.get_previous_transaction()).collect()
    }

    pub fn mutable_inputs(&self) -> BTreeMap<ObjectID, (VersionDigest, Owner)> {
        self.objects
            .iter()
            .filter_map(|ObjectReadResult { input_object_kind, object }| {
                match (input_object_kind, object) {
                    (
                        InputObjectKind::ImmOrOwnedObject(object_ref),
                        ObjectReadResultKind::Object(object),
                    ) => {
                        if object.is_immutable() {
                            None
                        } else {
                            Some((
                                object_ref.0,
                                ((object_ref.1, object_ref.2), object.owner.clone()),
                            ))
                        }
                    }
                    (
                        InputObjectKind::ImmOrOwnedObject(_),
                        ObjectReadResultKind::DeletedSharedObject(_, _),
                    ) => {
                        unreachable!()
                    }
                    (
                        InputObjectKind::SharedObject { .. },
                        ObjectReadResultKind::DeletedSharedObject(_, _),
                    ) => None,
                    (
                        InputObjectKind::SharedObject { mutable, .. },
                        ObjectReadResultKind::Object(object),
                    ) => {
                        if *mutable {
                            let oref = object.compute_object_reference();
                            Some((oref.0, ((oref.1, oref.2), object.owner.clone())))
                        } else {
                            None
                        }
                    }
                    (
                        InputObjectKind::ImmOrOwnedObject(_),
                        ObjectReadResultKind::CancelledTransactionSharedObject(_),
                    ) => {
                        unreachable!()
                    }
                    (
                        InputObjectKind::SharedObject { .. },
                        ObjectReadResultKind::CancelledTransactionSharedObject(_),
                    ) => None,
                }
            })
            .collect()
    }

    /// The version to set on objects created by the computation that `self` is input to.
    /// Guaranteed to be strictly greater than the versions of all input objects and objects
    /// received in the transaction.
    pub fn lamport_timestamp(&self, receiving_objects: &[ObjectRef]) -> Version {
        let input_versions = self
            .objects
            .iter()
            .filter_map(|object| match &object.object {
                ObjectReadResultKind::Object(object) => Some(object.data.version()),
                ObjectReadResultKind::DeletedSharedObject(v, _) => Some(*v),
                ObjectReadResultKind::CancelledTransactionSharedObject(_) => None,
            })
            .chain(receiving_objects.iter().map(|object_ref| object_ref.1));

        Version::lamport_increment(input_versions)
    }

    pub fn object_kinds(&self) -> impl Iterator<Item = &InputObjectKind> {
        self.objects.iter().map(|ObjectReadResult { input_object_kind, .. }| input_object_kind)
    }

    pub fn deleted_consensus_objects(&self) -> BTreeMap<ObjectID, Version> {
        self.objects
            .iter()
            .filter_map(|obj| {
                if let InputObjectKind::SharedObject { id, initial_shared_version, .. } =
                    obj.input_object_kind
                {
                    obj.is_deleted_shared_object().then_some((id, initial_shared_version))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn into_object_map(self) -> BTreeMap<ObjectID, Object> {
        self.objects
            .into_iter()
            .filter_map(|o| o.as_object().map(|object| (o.id(), object.clone())))
            .collect()
    }

    pub fn push(&mut self, object: ObjectReadResult) {
        self.objects.push(object);
    }

    pub fn iter(&self) -> impl Iterator<Item = &ObjectReadResult> {
        self.objects.iter()
    }

    pub fn iter_objects(&self) -> impl Iterator<Item = &Object> {
        self.objects.iter().filter_map(|o| o.as_object())
    }
}

// Result of attempting to read a receiving object (currently only at signing time).
// Because an object may have been previously received and deleted, the result may be
// ReceivingObjectReadResultKind::PreviouslyReceivedObject.
#[derive(Clone, Debug)]
pub enum ReceivingObjectReadResultKind {
    Object(Object),
    // The object was received by some other transaction, and we were not able to read it
    PreviouslyReceivedObject,
}

impl ReceivingObjectReadResultKind {
    pub fn as_object(&self) -> Option<&Object> {
        match &self {
            Self::Object(object) => Some(object),
            Self::PreviouslyReceivedObject => None,
        }
    }
}

pub struct ReceivingObjectReadResult {
    pub object_ref: ObjectRef,
    pub object: ReceivingObjectReadResultKind,
}

impl ReceivingObjectReadResult {
    pub fn new(object_ref: ObjectRef, object: ReceivingObjectReadResultKind) -> Self {
        Self { object_ref, object }
    }

    pub fn is_previously_received(&self) -> bool {
        matches!(self.object, ReceivingObjectReadResultKind::PreviouslyReceivedObject)
    }
}

impl From<Object> for ReceivingObjectReadResultKind {
    fn from(object: Object) -> Self {
        Self::Object(object)
    }
}

pub struct ReceivingObjects {
    pub objects: Vec<ReceivingObjectReadResult>,
}

impl ReceivingObjects {
    pub fn iter(&self) -> impl Iterator<Item = &ReceivingObjectReadResult> {
        self.objects.iter()
    }

    pub fn iter_objects(&self) -> impl Iterator<Item = &Object> {
        self.objects.iter().filter_map(|o| o.object.as_object())
    }
}

impl From<Vec<ReceivingObjectReadResult>> for ReceivingObjects {
    fn from(objects: Vec<ReceivingObjectReadResult>) -> Self {
        Self { objects }
    }
}
