use anyhow::anyhow;
use once_cell::sync::OnceCell;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    slice::Iter,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::{
    accumulator::CommitIndex,
    committee::{Committee, EpochId},
    crypto::{
        default_hash, AggregateAuthoritySignature, AuthoritySignInfo, AuthoritySignInfoTrait,
        AuthorityStrongQuorumSignInfo,
    },
    digests::{CommitContentsDigest, CommitSummaryDigest, TransactionDigest},
    envelope::{Envelope, Message, TrustedEnvelope, VerifiedEnvelope},
    error::{SomaError, SomaResult},
    intent::{Intent, IntentScope},
    storage::read_store::ReadStore,
    transaction::{Transaction, VerifiedTransaction},
};

pub type CommitTimestamp = u64;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CommitSummary {
    pub epoch: EpochId,
    pub index: CommitIndex,
    pub content_digest: CommitContentsDigest,
    pub previous_digest: Option<CommitSummaryDigest>,
    /// Timestamp of the commit - number of milliseconds from the Unix epoch
    /// Commit timestamps are monotonic, but not strongly monotonic - subsequent
    /// commits can have same timestamp if they originate from the same underlining consensus commit
    pub timestamp_ms: CommitTimestamp,
    // TODO: add state hash commit here potentially
    // TODO: add committee commitment here Vec(AuthorityName, StakeUnit)
}

impl Message for CommitSummary {
    type DigestType = CommitSummaryDigest;
    const SCOPE: IntentScope = IntentScope::CommitSummary;

    fn digest(&self) -> Self::DigestType {
        CommitSummaryDigest::new(default_hash(self))
    }
}

impl CommitSummary {
    pub fn new(
        epoch: EpochId,
        index: CommitIndex,
        transactions: &CommitContents,
        previous_digest: Option<CommitSummaryDigest>,
        // end_of_epoch_data: Option<EndOfEpochData>,
        timestamp_ms: CommitTimestamp,
    ) -> CommitSummary {
        let content_digest = *transactions.digest();

        Self {
            epoch,
            index,
            content_digest,
            previous_digest,
            timestamp_ms,
        }
    }

    pub fn verify_epoch(&self, epoch: EpochId) -> SomaResult {
        if !(self.epoch == epoch) {
            return Err(SomaError::WrongEpoch {
                expected_epoch: epoch,
                actual_epoch: self.epoch,
            });
        }

        Ok(())
    }

    pub fn index(&self) -> &CommitIndex {
        &self.index
    }

    pub fn timestamp(&self) -> SystemTime {
        UNIX_EPOCH + Duration::from_millis(self.timestamp_ms)
    }

    pub fn is_last_commit_of_epoch(&self) -> bool {
        // TODO:
        // self.end_of_epoch_data.is_some()
        false
    }

    // TODO: pub fn next_epoch_committee(&self) -> Option<&[(AuthorityName, StakeUnit)]> {
    //     self.end_of_epoch_data
    //         .as_ref()
    //         .map(|e| e.next_epoch_committee.as_slice())
    // }
}

impl Display for CommitSummary {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CheckpointSummary {{ epoch: {:?}, seq: {:?}, content_digest: {}}}",
            self.epoch, self.index, self.content_digest,
        )
    }
}

// Commit summaries are signed by an authority and 2f+1 form a
// certificate that others can use to catch up. The actual
// content of the digest must at the very least commit to
// the set of transactions contained in the certificate but
// we might extend this to contain roots of merkle trees,
// or other authenticated data structures to support light
// clients and more efficient sync protocols.

pub type CommitSummaryEnvelope<S> = Envelope<CommitSummary, S>;
pub type CertifiedCommitSummary = CommitSummaryEnvelope<AuthorityStrongQuorumSignInfo>;
pub type SignedCommitSummary = CommitSummaryEnvelope<AuthoritySignInfo>;

pub type VerifiedCommitSummary = VerifiedEnvelope<CommitSummary, AuthorityStrongQuorumSignInfo>;
pub type TrustedCommitSummary = TrustedEnvelope<CommitSummary, AuthorityStrongQuorumSignInfo>;

impl CertifiedCommitSummary {
    pub fn verify_authority_signatures(&self, committee: &Committee) -> SomaResult {
        self.data().verify_epoch(self.auth_sig().epoch)?;
        self.auth_sig().verify_secure(
            self.data(),
            Intent::consensus_app(IntentScope::CommitSummary),
            committee,
        )
    }

    pub fn try_into_verified(self, committee: &Committee) -> SomaResult<VerifiedCommitSummary> {
        self.verify_authority_signatures(committee)?;
        Ok(VerifiedCommitSummary::new_from_verified(self))
    }

    pub fn verify_with_contents(
        &self,
        committee: &Committee,
        contents: Option<&CommitContents>,
    ) -> SomaResult {
        self.verify_authority_signatures(committee)?;

        if let Some(contents) = contents {
            let content_digest = *contents.digest();
            if !(content_digest == self.data().content_digest) {
                return Err(SomaError::GenericAuthorityError {
                    error: format!(
                        "Commit contents digest mismatch: summary={:?}, received content digest {:?}, received {} transactions",
                        self.data(),
                        content_digest,
                        contents.size()
                    ),
                });
            }
        }

        Ok(())
    }

    pub fn into_summary_and_index(self) -> (CommitIndex, CommitSummary) {
        let summary = self.into_data();
        (summary.index, summary)
    }

    pub fn get_validator_signature(self) -> AggregateAuthoritySignature {
        self.auth_sig().signature.clone()
    }
}

impl SignedCommitSummary {
    pub fn verify_authority_signatures(&self, committee: &Committee) -> SomaResult {
        self.data().verify_epoch(self.auth_sig().epoch)?;
        self.auth_sig().verify_secure(
            self.data(),
            Intent::consensus_app(IntentScope::CommitSummary),
            committee,
        )
    }

    pub fn try_into_verified(
        self,
        committee: &Committee,
    ) -> SomaResult<VerifiedEnvelope<CommitSummary, AuthoritySignInfo>> {
        self.verify_authority_signatures(committee)?;
        Ok(VerifiedEnvelope::<CommitSummary, AuthoritySignInfo>::new_from_verified(self))
    }
}

impl VerifiedCommitSummary {
    pub fn into_summary_and_index(self) -> (CommitIndex, CommitSummary) {
        self.into_inner().into_summary_and_index()
    }
}

/// CommitContents are the transactions included in a commit.
/// They must have already been causally ordered. Since the causal order algorithm
/// is the same among validators, we expect all honest validators to come up with
/// the same order for each commit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CommitContents {
    #[serde(skip)]
    digest: OnceCell<CommitContentsDigest>,

    transactions: Vec<TransactionDigest>,
}

impl CommitContents {
    pub fn new_with_digests<T>(contents: T) -> Self
    where
        T: IntoIterator<Item = TransactionDigest>,
    {
        let transactions: Vec<_> = contents.into_iter().collect();

        CommitContents {
            digest: Default::default(),
            transactions,
        }
    }

    pub fn new_with_causally_ordered_transactions<'a, T>(contents: T) -> Self
    where
        T: IntoIterator<Item = &'a VerifiedTransaction>,
    {
        let transactions: Vec<_> = contents.into_iter().map(|data| *data.digest()).collect();

        CommitContents {
            digest: Default::default(),
            transactions,
        }
    }

    pub fn iter(&self) -> Iter<'_, TransactionDigest> {
        self.transactions.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = TransactionDigest> {
        self.transactions.into_iter()
    }

    pub fn into_inner(self) -> Vec<TransactionDigest> {
        self.transactions
    }

    pub fn inner(&self) -> &[TransactionDigest] {
        &self.transactions
    }

    pub fn size(&self) -> usize {
        self.transactions.len()
    }

    pub fn digest(&self) -> &CommitContentsDigest {
        self.digest
            .get_or_init(|| CommitContentsDigest::new(default_hash(self)))
    }
}

/// Same as CommitContents, but contains full contents of all Transactions and
/// associated with the commit.
// NOTE: This data structure is used for state sync of commits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullCommitContents {
    transactions: Vec<Transaction>,
}

impl FullCommitContents {
    pub fn new_with_causally_ordered_transactions<T>(contents: T) -> Self
    where
        T: IntoIterator<Item = Transaction>,
    {
        Self {
            transactions: contents.into_iter().collect(),
        }
    }
    pub fn from_contents_and_transactions(
        contents: CommitContents,
        transactions: impl Iterator<Item = Transaction>,
    ) -> Self {
        let transactions: Vec<_> = transactions.collect();
        Self { transactions }
    }

    pub fn from_commit_contents<S>(store: S, contents: CommitContents) -> Option<Self>
    where
        S: ReadStore,
    {
        let mut transactions = Vec::with_capacity(contents.size());
        for tx in contents.iter() {
            if let Ok(Some(t)) = store.get_transaction(&tx) {
                transactions.push(Transaction::new((*t).data().clone()))
            } else {
                return None;
            }
        }
        Some(Self { transactions })
    }

    pub fn iter(&self) -> Iter<'_, Transaction> {
        self.transactions.iter()
    }

    /// Verifies that this commit's digest matches the given digest.
    pub fn verify_digests(&self, digest: CommitContentsDigest) -> anyhow::Result<()> {
        let self_digest = *self.commit_contents().digest();
        if !(digest == self_digest) {
            return Err(anyhow!(
                "commit contents digest {self_digest} does not match expected digest {digest}"
            ));
        }

        Ok(())
    }

    pub fn commit_contents(&self) -> CommitContents {
        CommitContents {
            digest: Default::default(),
            transactions: self.transactions.iter().map(|tx| *tx.digest()).collect(),
        }
    }

    pub fn into_commit_contents(self) -> CommitContents {
        CommitContents {
            digest: Default::default(),
            transactions: self
                .transactions
                .into_iter()
                .map(|tx| *tx.digest())
                .collect(),
        }
    }

    pub fn size(&self) -> usize {
        self.transactions.len()
    }
}

impl IntoIterator for FullCommitContents {
    type Item = Transaction;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.transactions.into_iter()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedCommitContents {
    transactions: Vec<VerifiedTransaction>,
}

impl VerifiedCommitContents {
    pub fn new_unchecked(contents: FullCommitContents) -> Self {
        Self {
            transactions: contents
                .transactions
                .into_iter()
                .map(VerifiedTransaction::new_unchecked)
                .collect(),
        }
    }

    pub fn iter(&self) -> Iter<'_, VerifiedTransaction> {
        self.transactions.iter()
    }

    pub fn transactions(&self) -> &[VerifiedTransaction] {
        &self.transactions
    }

    pub fn into_inner(self) -> FullCommitContents {
        FullCommitContents {
            transactions: self
                .transactions
                .into_iter()
                .map(|tx| tx.into_inner())
                .collect(),
        }
    }

    pub fn into_checkpoint_contents(self) -> CommitContents {
        self.into_inner().into_commit_contents()
    }

    pub fn into_commit_contents_digest(self) -> CommitContentsDigest {
        *self.into_inner().into_commit_contents().digest()
    }

    pub fn num_of_transactions(&self) -> usize {
        self.transactions.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GetCommitSummaryRequest {
    Latest,
    ByDigest(CommitSummaryDigest),
    ByIndex(CommitIndex),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetCommitAvailabilityResponse {
    pub highest_synced_commit: CertifiedCommitSummary,
    pub lowest_available_commit: CommitIndex,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PushCommitSummaryResponse {
    pub timestamp_ms: CommitTimestamp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetCommitAvailabilityRequest {
    pub timestamp_ms: CommitTimestamp,
}
