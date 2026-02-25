// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::slice::Iter;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use fastcrypto::hash::{Blake2b256, MultisetHash as _};
use fastcrypto::merkle::{MerkleProof, MerkleTree, Node};
use once_cell::sync::OnceCell;
use protocol_config::ProtocolVersion;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::base::{
    AuthorityName, ExecutionData, ExecutionDigests, FullObjectRef, SequenceNumber,
    VerifiedExecutionData,
};
use crate::committee::{Committee, EpochId, StakeUnit};
use crate::crypto::{
    AggregateAuthoritySignature, AuthoritySignInfo, AuthoritySignInfoTrait as _,
    AuthorityStrongQuorumSignInfo, GenericSignature, SomaKeyPair, default_hash, get_key_pair,
};
use crate::digests::{
    CheckpointArtifactsDigest, CheckpointContentsDigest, CheckpointDigest, Digest, ObjectDigest,
    TransactionDigest, TransactionEffectsDigest,
};
use crate::effects::{TransactionEffects, TransactionEffectsAPI as _};
use crate::envelope::{Envelope, Message, TrustedEnvelope, VerifiedEnvelope};
use crate::error::{SomaError, SomaResult};
use crate::full_checkpoint_content::CheckpointData;
use crate::intent::{Intent, IntentScope};
use crate::object::{ObjectID, Version};
use crate::serde::Readable;
use crate::transaction::{Transaction, TransactionData};
use crate::tx_fee::TransactionFee;

pub type CheckpointSequenceNumber = u64;
pub type CheckpointTimestamp = u64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointRequest {
    /// if a sequence number is specified, return the checkpoint with that sequence number;
    /// otherwise if None returns the latest checkpoint stored (authenticated or pending,
    /// depending on the value of `certified` flag)
    pub sequence_number: Option<CheckpointSequenceNumber>,
    // A flag, if true also return the contents of the
    // checkpoint besides the meta-data.
    pub request_content: bool,
    // If true, returns certified checkpoint, otherwise returns pending checkpoint
    pub certified: bool,
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CheckpointSummaryResponse {
    Certified(CertifiedCheckpointSummary),
    Pending(CheckpointSummary),
}

impl CheckpointSummaryResponse {
    pub fn content_digest(&self) -> CheckpointContentsDigest {
        match self {
            Self::Certified(s) => s.content_digest,
            Self::Pending(s) => s.content_digest,
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointResponse {
    pub checkpoint: Option<CheckpointSummaryResponse>,
    pub contents: Option<CheckpointContents>,
}

pub type GlobalStateHash = fastcrypto::hash::EllipticCurveMultisetHash;

/// The Sha256 digest of an EllipticCurveMultisetHash committing to the live object set.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ECMHLiveObjectSetDigest {
    #[schemars(with = "[u8; 32]")]
    pub digest: Digest,
}

impl From<fastcrypto::hash::Digest<32>> for ECMHLiveObjectSetDigest {
    fn from(digest: fastcrypto::hash::Digest<32>) -> Self {
        Self { digest: Digest::new(digest.digest) }
    }
}

impl Default for ECMHLiveObjectSetDigest {
    fn default() -> Self {
        GlobalStateHash::default().digest().into()
    }
}

/// CheckpointArtifact is a type that represents various artifacts of a checkpoint.
/// We hash all the artifacts together to get the checkpoint artifacts digest
/// that is included in the checkpoint summary.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CheckpointArtifact {
    /// The post-checkpoint state of all objects modified in the checkpoint.
    /// It also includes objects that were deleted or wrapped in the checkpoint.
    ObjectStates(BTreeMap<ObjectID, (Version, ObjectDigest)>),
    // In the future, we can add more artifacts e.g., execution digests, etc.
}

impl CheckpointArtifact {
    pub fn digest(&self) -> SomaResult<Digest> {
        match self {
            Self::ObjectStates(object_states) => {
                let tree = MerkleTree::<Blake2b256>::build_from_unserialized(
                    object_states.iter().map(|(id, (seq, digest))| (id, seq, digest)),
                )
                .map_err(|e| SomaError::GenericAuthorityError {
                    error: format!("Failed to build Merkle tree: {}", e),
                })?;
                let root = tree.root().bytes();
                Ok(Digest::new(root))
            }
        }
    }

    pub fn artifact_type(&self) -> &'static str {
        match self {
            Self::ObjectStates(_) => "ObjectStates",
            // Future variants...
        }
    }
}

#[derive(Debug)]
pub struct CheckpointArtifacts {
    /// An ordered list of artifacts.
    artifacts: BTreeSet<CheckpointArtifact>,
}

impl CheckpointArtifacts {
    pub fn new() -> Self {
        Self { artifacts: BTreeSet::new() }
    }

    pub fn add_artifact(&mut self, artifact: CheckpointArtifact) -> SomaResult<()> {
        if self
            .artifacts
            .iter()
            .any(|existing| existing.artifact_type() == artifact.artifact_type())
        {
            return Err(SomaError::GenericAuthorityError {
                error: format!("Artifact {} already exists", artifact.artifact_type()),
            });
        }
        self.artifacts.insert(artifact);
        Ok(())
    }

    pub fn from_object_states(object_states: BTreeMap<ObjectID, (Version, ObjectDigest)>) -> Self {
        CheckpointArtifacts {
            artifacts: BTreeSet::from([CheckpointArtifact::ObjectStates(object_states)]),
        }
    }

    /// Get the object states if present
    pub fn object_states(&self) -> SomaResult<&BTreeMap<ObjectID, (Version, ObjectDigest)>> {
        self.artifacts
            .iter()
            .find(|artifact| matches!(artifact, CheckpointArtifact::ObjectStates(_)))
            .map(|artifact| match artifact {
                CheckpointArtifact::ObjectStates(states) => states,
            })
            .ok_or(SomaError::GenericAuthorityError {
                error: "Object states not found in checkpoint artifacts".to_string(),
            })
    }

    pub fn digest(&self) -> SomaResult<CheckpointArtifactsDigest> {
        // Already sorted by BTreeSet!
        let digests = self.artifacts.iter().map(|a| a.digest()).collect::<Result<Vec<_>, _>>()?;

        CheckpointArtifactsDigest::from_artifact_digests(digests)
    }
}

impl Default for CheckpointArtifacts {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&[&TransactionEffects]> for CheckpointArtifacts {
    fn from(effects: &[&TransactionEffects]) -> Self {
        let mut latest_object_states = BTreeMap::new();
        for e in effects {
            for (id, seq, digest) in e.written() {
                if let Some((old_seq, _)) = latest_object_states.insert(id, (seq, digest)) {
                    assert!(old_seq < seq, "Object states should be monotonically increasing");
                }
            }
        }

        CheckpointArtifacts::from_object_states(latest_object_states)
    }
}

impl From<&[TransactionEffects]> for CheckpointArtifacts {
    fn from(effects: &[TransactionEffects]) -> Self {
        let effect_refs: Vec<&TransactionEffects> = effects.iter().collect();
        Self::from(effect_refs.as_slice())
    }
}

impl From<&CheckpointData> for CheckpointArtifacts {
    fn from(checkpoint_data: &CheckpointData) -> Self {
        let effects = checkpoint_data.transactions.iter().map(|tx| &tx.effects).collect::<Vec<_>>();

        Self::from(effects.as_slice())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub enum CheckpointCommitment {
    ECMHLiveObjectSetDigest(ECMHLiveObjectSetDigest),
    CheckpointArtifactsDigest(CheckpointArtifactsDigest),
}

impl From<ECMHLiveObjectSetDigest> for CheckpointCommitment {
    fn from(d: ECMHLiveObjectSetDigest) -> Self {
        Self::ECMHLiveObjectSetDigest(d)
    }
}

impl From<CheckpointArtifactsDigest> for CheckpointCommitment {
    fn from(d: CheckpointArtifactsDigest) -> Self {
        Self::CheckpointArtifactsDigest(d)
    }
}

// #[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct EndOfEpochData {
    /// next_epoch_committee is `Some` if and only if the current checkpoint is
    /// the last checkpoint of an epoch.
    /// Therefore next_epoch_committee can be used to pick the last checkpoint of an epoch,
    /// which is often useful to get epoch level summary stats like total gas cost of an epoch,
    /// or the total number of transactions from genesis to the end of an epoch.

    /// The validator committee for the next epoch, including network metadata
    pub next_epoch_validator_committee: Committee,

    /// The protocol version that is in effect during the epoch that starts immediately after this
    ///checkpoint.
    pub next_epoch_protocol_version: ProtocolVersion,

    /// Commitments to epoch specific state (e.g. live object set)
    pub epoch_commitments: Vec<CheckpointCommitment>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointSummary {
    pub epoch: EpochId,
    pub sequence_number: CheckpointSequenceNumber,
    /// Total number of transactions committed since genesis, including those in this
    /// checkpoint.
    pub network_total_transactions: u64,
    pub content_digest: CheckpointContentsDigest,
    pub previous_digest: Option<CheckpointDigest>,
    /// The running total fees of all transactions included in the current epoch so far
    /// until this checkpoint.
    pub epoch_rolling_transaction_fees: TransactionFee,

    /// Timestamp of the checkpoint - number of milliseconds from the Unix epoch
    /// Checkpoint timestamps are monotonic, but not strongly monotonic - subsequent
    /// checkpoints can have same timestamp if they originate from the same underlining consensus commit
    pub timestamp_ms: CheckpointTimestamp,

    /// Commitments to checkpoint-specific state (e.g. txns in checkpoint, objects read/written in
    /// checkpoint).
    pub checkpoint_commitments: Vec<CheckpointCommitment>,

    /// Present only on the final checkpoint of the epoch.
    pub end_of_epoch_data: Option<EndOfEpochData>,

    /// Opaque version-specific data for forward compatibility.
    /// Protocol version upgrades can store additional data here without changing the struct layout.
    #[serde(default)]
    pub version_specific_data: Vec<u8>,
}

impl Message for CheckpointSummary {
    type DigestType = CheckpointDigest;
    const SCOPE: IntentScope = IntentScope::CheckpointSummary;

    fn digest(&self) -> Self::DigestType {
        CheckpointDigest::new(default_hash(self))
    }
}

impl CheckpointSummary {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        epoch: EpochId,
        sequence_number: CheckpointSequenceNumber,
        network_total_transactions: u64,
        transactions: &CheckpointContents,
        previous_digest: Option<CheckpointDigest>,
        epoch_rolling_transaction_fees: TransactionFee,
        end_of_epoch_data: Option<EndOfEpochData>,
        timestamp_ms: CheckpointTimestamp,
        checkpoint_commitments: Vec<CheckpointCommitment>,
    ) -> CheckpointSummary {
        let content_digest = *transactions.digest();

        Self {
            epoch,
            sequence_number,
            network_total_transactions,
            content_digest,
            previous_digest,
            epoch_rolling_transaction_fees,
            end_of_epoch_data,
            timestamp_ms,
            checkpoint_commitments,
            version_specific_data: Vec::new(),
        }
    }

    pub fn verify_epoch(&self, epoch: EpochId) -> SomaResult {
        if self.epoch != epoch {
            return Err(SomaError::WrongEpoch { expected_epoch: epoch, actual_epoch: self.epoch });
        }

        Ok(())
    }

    pub fn sequence_number(&self) -> &CheckpointSequenceNumber {
        &self.sequence_number
    }

    pub fn timestamp(&self) -> SystemTime {
        UNIX_EPOCH + Duration::from_millis(self.timestamp_ms)
    }

    pub fn next_epoch_committee(&self) -> Option<&Committee> {
        self.end_of_epoch_data.as_ref().map(|e| &e.next_epoch_validator_committee)
    }

    pub fn is_last_checkpoint_of_epoch(&self) -> bool {
        self.end_of_epoch_data.is_some()
    }

    pub fn checkpoint_artifacts_digest(&self) -> SomaResult<&CheckpointArtifactsDigest> {
        self.checkpoint_commitments
            .iter()
            .find_map(|c| match c {
                CheckpointCommitment::CheckpointArtifactsDigest(digest) => Some(digest),
                _ => None,
            })
            .ok_or(SomaError::GenericAuthorityError {
                error: "Checkpoint artifacts digest not found in checkpoint commitments"
                    .to_string(),
            })
    }
}

impl Display for CheckpointSummary {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CheckpointSummary {{ epoch: {:?}, seq: {:?}, content_digest: {},
            epoch_rolling_transaction_fees: {:?}}}",
            self.epoch,
            self.sequence_number,
            self.content_digest,
            self.epoch_rolling_transaction_fees,
        )
    }
}

// Checkpoints are signed by an authority and 2f+1 form a
// certificate that others can use to catch up. The actual
// content of the digest must at the very least commit to
// the set of transactions contained in the certificate but
// we might extend this to contain roots of merkle trees,
// or other authenticated data structures to support light
// clients and more efficient sync protocols.

pub type CheckpointSummaryEnvelope<S> = Envelope<CheckpointSummary, S>;
pub type CertifiedCheckpointSummary = CheckpointSummaryEnvelope<AuthorityStrongQuorumSignInfo>;
pub type SignedCheckpointSummary = CheckpointSummaryEnvelope<AuthoritySignInfo>;

pub type VerifiedCheckpoint = VerifiedEnvelope<CheckpointSummary, AuthorityStrongQuorumSignInfo>;
pub type TrustedCheckpoint = TrustedEnvelope<CheckpointSummary, AuthorityStrongQuorumSignInfo>;

impl CertifiedCheckpointSummary {
    pub fn verify_authority_signatures(&self, committee: &Committee) -> SomaResult {
        self.data().verify_epoch(self.auth_sig().epoch)?;
        self.auth_sig().verify_secure(
            self.data(),
            Intent::soma_app(IntentScope::CheckpointSummary),
            committee,
        )
    }

    pub fn try_into_verified(self, committee: &Committee) -> SomaResult<VerifiedCheckpoint> {
        self.verify_authority_signatures(committee)?;
        Ok(VerifiedCheckpoint::new_from_verified(self))
    }

    pub fn verify_with_contents(
        &self,
        committee: &Committee,
        contents: Option<&CheckpointContents>,
    ) -> SomaResult {
        self.verify_authority_signatures(committee)?;

        if let Some(contents) = contents {
            let content_digest = *contents.digest();
            if content_digest != self.data().content_digest {
                return Err(SomaError::GenericAuthorityError {
                    error: format!(
                        "Checkpoint contents digest mismatch: summary={:?}, received content digest {:?}, received {} transactions",
                        self.data(),
                        content_digest,
                        contents.size()
                    ),
                });
            }
        }

        Ok(())
    }

    pub fn into_summary_and_sequence(self) -> (CheckpointSequenceNumber, CheckpointSummary) {
        let summary = self.into_data();
        (summary.sequence_number, summary)
    }

    pub fn get_validator_signature(self) -> AggregateAuthoritySignature {
        self.auth_sig().signature.clone()
    }
}

impl SignedCheckpointSummary {
    pub fn verify_authority_signatures(&self, committee: &Committee) -> SomaResult {
        self.data().verify_epoch(self.auth_sig().epoch)?;
        self.auth_sig().verify_secure(
            self.data(),
            Intent::soma_app(IntentScope::CheckpointSummary),
            committee,
        )
    }

    pub fn try_into_verified(
        self,
        committee: &Committee,
    ) -> SomaResult<VerifiedEnvelope<CheckpointSummary, AuthoritySignInfo>> {
        self.verify_authority_signatures(committee)?;
        Ok(VerifiedEnvelope::<CheckpointSummary, AuthoritySignInfo>::new_from_verified(self))
    }
}

impl VerifiedCheckpoint {
    pub fn into_summary_and_sequence(self) -> (CheckpointSequenceNumber, CheckpointSummary) {
        self.into_inner().into_summary_and_sequence()
    }
}

/// This is a message validators publish to consensus in order to sign checkpoint
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointSignatureMessage {
    pub summary: SignedCheckpointSummary,
}

impl CheckpointSignatureMessage {
    pub fn verify(&self, committee: &Committee) -> SomaResult {
        self.summary.verify_authority_signatures(committee)
    }
}

/// A proof that a specific ExecutionDigests is included in a checkpoint's contents.
/// Size is O(log n) instead of O(n) where n = number of transactions in checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInclusionProof {
    /// The leaf being proven (the ExecutionDigests for our transaction)
    pub leaf: ExecutionDigests,
    /// The index of this leaf in the checkpoint contents
    pub leaf_index: usize,
    /// The Merkle proof (sibling hashes along path from leaf to root)
    pub proof: MerkleProof<Blake2b256>,
}

impl CheckpointInclusionProof {
    /// Verify this proof against an expected content digest (Merkle root)
    pub fn verify(&self, expected_root: &CheckpointContentsDigest) -> SomaResult<()> {
        let root_node = Node::Digest(expected_root.into_inner());

        self.proof
            .verify_proof_with_unserialized_leaf(&root_node, &self.leaf, self.leaf_index)
            .map_err(|e| {
                SomaError::InvalidFinalityProof(format!(
                    "Merkle proof verification failed: {:?}",
                    e
                ))
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointContents {
    V1(CheckpointContentsV1),
}

/// CheckpointContents are the transactions included in an upcoming checkpoint.
/// They must have already been causally ordered. Since the causal order algorithm
/// is the same among validators, we expect all honest validators to come up with
/// the same order for each checkpoint content.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointContentsV1 {
    #[serde(skip)]
    digest: OnceCell<CheckpointContentsDigest>,

    transactions: Vec<ExecutionDigests>,
    /// This field 'pins' user signatures for the checkpoint
    /// The length of this vector is same as length of transactions vector
    /// System transactions has empty signatures
    user_signatures: Vec<Vec<GenericSignature>>,
}

impl CheckpointContents {
    pub fn new_with_digests_and_signatures<T>(
        contents: T,
        user_signatures: Vec<Vec<GenericSignature>>,
    ) -> Self
    where
        T: IntoIterator<Item = ExecutionDigests>,
    {
        let transactions: Vec<_> = contents.into_iter().collect();
        assert_eq!(transactions.len(), user_signatures.len());
        Self::V1(CheckpointContentsV1 { digest: Default::default(), transactions, user_signatures })
    }

    pub fn new_with_causally_ordered_execution_data<'a, T>(contents: T) -> Self
    where
        T: IntoIterator<Item = &'a VerifiedExecutionData>,
    {
        let (transactions, user_signatures): (Vec<_>, Vec<_>) = contents
            .into_iter()
            .map(|data| {
                (data.digests(), data.transaction.inner().data().tx_signatures().to_owned())
            })
            .unzip();
        assert_eq!(transactions.len(), user_signatures.len());
        Self::V1(CheckpointContentsV1 { digest: Default::default(), transactions, user_signatures })
    }

    pub fn new_with_digests_only_for_tests<T>(contents: T) -> Self
    where
        T: IntoIterator<Item = ExecutionDigests>,
    {
        let transactions: Vec<_> = contents.into_iter().collect();
        let user_signatures = transactions.iter().map(|_| vec![]).collect();
        Self::V1(CheckpointContentsV1 { digest: Default::default(), transactions, user_signatures })
    }

    fn as_v1(&self) -> &CheckpointContentsV1 {
        match self {
            Self::V1(v) => v,
        }
    }

    fn into_v1(self) -> CheckpointContentsV1 {
        match self {
            Self::V1(v) => v,
        }
    }

    pub fn iter(&self) -> Iter<'_, ExecutionDigests> {
        self.as_v1().transactions.iter()
    }

    pub fn into_iter_with_signatures(
        self,
    ) -> impl Iterator<Item = (ExecutionDigests, Vec<GenericSignature>)> {
        let CheckpointContentsV1 { transactions, user_signatures, .. } = self.into_v1();

        transactions.into_iter().zip(user_signatures)
    }

    /// Return an iterator that enumerates the transactions in the contents.
    /// The iterator item is a tuple of (sequence_number, &ExecutionDigests),
    /// where the sequence_number indicates the index of the transaction in the
    /// global ordering of executed transactions since genesis.
    pub fn enumerate_transactions(
        &self,
        ckpt: &CheckpointSummary,
    ) -> impl Iterator<Item = (u64, &ExecutionDigests)> {
        let start = ckpt.network_total_transactions - self.size() as u64;

        (0u64..).zip(self.iter()).map(move |(i, digests)| (i + start, digests))
    }

    pub fn into_inner(self) -> Vec<ExecutionDigests> {
        self.into_v1().transactions
    }

    pub fn inner(&self) -> &[ExecutionDigests] {
        &self.as_v1().transactions
    }

    pub fn size(&self) -> usize {
        self.as_v1().transactions.len()
    }

    /// Returns the Merkle root of the checkpoint contents.
    /// Cached for efficiency.
    pub fn digest(&self) -> &CheckpointContentsDigest {
        self.as_v1().digest.get_or_init(|| {
            // Build Merkle tree and use root as digest
            let leaves: Vec<_> = self.iter().cloned().collect();

            if leaves.is_empty() {
                return CheckpointContentsDigest::new(fastcrypto::merkle::EMPTY_NODE);
            }

            let tree = MerkleTree::<Blake2b256>::build_from_unserialized(leaves.iter())
                .expect("Failed to build Merkle tree for checkpoint contents");

            CheckpointContentsDigest::new(tree.root().bytes())
        })
    }

    /// Build a Merkle tree over the execution digests and return the root as the digest.
    /// This replaces the old hash-based digest computation.
    pub fn compute_digest(&self) -> SomaResult<CheckpointContentsDigest> {
        let leaves: Vec<_> = self.iter().cloned().collect();

        if leaves.is_empty() {
            // Empty checkpoint - use empty node hash
            return Ok(CheckpointContentsDigest::new(fastcrypto::merkle::EMPTY_NODE));
        }

        let tree =
            MerkleTree::<Blake2b256>::build_from_unserialized(leaves.iter()).map_err(|e| {
                SomaError::GenericAuthorityError {
                    error: format!("Failed to build Merkle tree: {:?}", e),
                }
            })?;

        Ok(CheckpointContentsDigest::new(tree.root().bytes()))
    }

    /// Generate an inclusion proof for a specific transaction.
    /// Returns the proof that can be verified against the checkpoint's content_digest.
    pub fn generate_inclusion_proof(
        &self,
        tx_digest: &TransactionDigest,
        effects_digest: &TransactionEffectsDigest,
    ) -> SomaResult<CheckpointInclusionProof> {
        let leaves: Vec<_> = self.iter().cloned().collect();

        // Find the leaf index
        let leaf_index = leaves
            .iter()
            .position(|ed| &ed.transaction == tx_digest && &ed.effects == effects_digest)
            .ok_or_else(|| {
                SomaError::InvalidFinalityProof(
                    "Transaction not found in checkpoint contents".to_string(),
                )
            })?;

        let tree =
            MerkleTree::<Blake2b256>::build_from_unserialized(leaves.iter()).map_err(|e| {
                SomaError::GenericAuthorityError {
                    error: format!("Failed to build Merkle tree: {:?}", e),
                }
            })?;

        let proof = tree.get_proof(leaf_index).map_err(|e| SomaError::GenericAuthorityError {
            error: format!("Failed to generate Merkle proof: {:?}", e),
        })?;

        Ok(CheckpointInclusionProof { leaf: leaves[leaf_index], leaf_index, proof })
    }
}

/// Same as CheckpointContents, but contains full contents of all Transactions and
/// TransactionEffects associated with the checkpoint.
// NOTE: This data structure is used for state sync of checkpoints. Therefore we attempt
// to estimate its size in CheckpointBuilder in order to limit the maximum serialized
// size of a checkpoint sent over the network. If this struct is modified,
// CheckpointBuilder::split_checkpoint_chunks should also be updated accordingly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FullCheckpointContents {
    transactions: Vec<ExecutionData>,
    /// This field 'pins' user signatures for the checkpoint
    /// The length of this vector is same as length of transactions vector
    /// System transactions has empty signatures
    user_signatures: Vec<Vec<GenericSignature>>,
}

impl FullCheckpointContents {
    pub fn new_with_causally_ordered_transactions<T>(contents: T) -> Self
    where
        T: IntoIterator<Item = ExecutionData>,
    {
        let (transactions, user_signatures): (Vec<_>, Vec<_>) = contents
            .into_iter()
            .map(|data| {
                let sig = data.transaction.data().tx_signatures().to_owned();
                (data, sig)
            })
            .unzip();
        assert_eq!(transactions.len(), user_signatures.len());
        Self { transactions, user_signatures }
    }
    pub fn from_contents_and_execution_data(
        contents: CheckpointContents,
        execution_data: impl Iterator<Item = ExecutionData>,
    ) -> Self {
        let transactions: Vec<_> = execution_data.collect();
        Self { transactions, user_signatures: contents.into_v1().user_signatures }
    }

    pub fn iter(&self) -> Iter<'_, ExecutionData> {
        self.transactions.iter()
    }

    /// Verifies that this checkpoint's digest matches the given digest, and that all internal
    /// Transaction and TransactionEffects digests are consistent.
    pub fn verify_digests(&self, digest: CheckpointContentsDigest) -> Result<()> {
        let self_digest = *self.checkpoint_contents().digest();
        if digest != self_digest {
            return Err(anyhow::anyhow!(
                "checkpoint contents digest {self_digest} does not match expected digest {digest}"
            ));
        }
        for tx in self.iter() {
            let transaction_digest = tx.transaction.digest();
            if tx.effects.transaction_digest() != transaction_digest {
                return Err(anyhow::anyhow!(
                    "transaction digest {transaction_digest} does not match expected digest {}",
                    tx.effects.transaction_digest()
                ));
            }
        }
        Ok(())
    }

    pub fn checkpoint_contents(&self) -> CheckpointContents {
        CheckpointContents::V1(CheckpointContentsV1 {
            digest: Default::default(),
            transactions: self.transactions.iter().map(|tx| tx.digests()).collect(),
            user_signatures: self.user_signatures.clone(),
        })
    }

    pub fn into_checkpoint_contents(self) -> CheckpointContents {
        CheckpointContents::V1(CheckpointContentsV1 {
            digest: Default::default(),
            transactions: self.transactions.into_iter().map(|tx| tx.digests()).collect(),
            user_signatures: self.user_signatures,
        })
    }

    pub fn size(&self) -> usize {
        self.transactions.len()
    }
}

impl IntoIterator for FullCheckpointContents {
    type Item = ExecutionData;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.transactions.into_iter()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedCheckpointContents {
    transactions: Vec<VerifiedExecutionData>,
    /// This field 'pins' user signatures for the checkpoint
    /// The length of this vector is same as length of transactions vector
    /// System transactions has empty signatures
    user_signatures: Vec<Vec<GenericSignature>>,
}

impl VerifiedCheckpointContents {
    pub fn new_unchecked(contents: FullCheckpointContents) -> Self {
        Self {
            transactions: contents
                .transactions
                .into_iter()
                .map(VerifiedExecutionData::new_unchecked)
                .collect(),
            user_signatures: contents.user_signatures,
        }
    }

    pub fn iter(&self) -> Iter<'_, VerifiedExecutionData> {
        self.transactions.iter()
    }

    pub fn transactions(&self) -> &[VerifiedExecutionData] {
        &self.transactions
    }

    pub fn into_inner(self) -> FullCheckpointContents {
        FullCheckpointContents {
            transactions: self.transactions.into_iter().map(|tx| tx.into_inner()).collect(),
            user_signatures: self.user_signatures,
        }
    }

    pub fn into_checkpoint_contents(self) -> CheckpointContents {
        self.into_inner().into_checkpoint_contents()
    }

    pub fn into_checkpoint_contents_digest(self) -> CheckpointContentsDigest {
        *self.into_inner().into_checkpoint_contents().digest()
    }

    pub fn num_of_transactions(&self) -> usize {
        self.transactions.len()
    }
}
