use async_trait::async_trait;
use data_ingestion::Worker;
use std::sync::Arc;
use tracing::{debug, info};
use types::{
    checkpoint::CommitArchiveData,
    consensus::{
        block::{BlockAPI, SignedBlock, VerifiedBlock},
        block_verifier::{BlockVerifier as _, SignedBlockVerifier},
        commit::{Commit, CommitAPI, CommittedSubDag, TrustedCommit},
        stake_aggregator::{QuorumThreshold, StakeAggregator},
    },
    error::{ConsensusError, SomaError},
    storage::{
        consensus::{ConsensusStore, WriteBatch},
        write_store::WriteStore,
    },
};
// use sui_data_ingestion_core::Worker;

pub struct StateSyncWorker<S> {
    pub store: S,
    pub block_verifier: Arc<SignedBlockVerifier>,
}

impl<S> StateSyncWorker<S> {
    pub fn new(store: S, block_verifier: Arc<SignedBlockVerifier>) -> Self {
        Self {
            store,
            block_verifier,
        }
    }
}

#[async_trait]
impl<S> Worker for StateSyncWorker<S>
where
    S: ConsensusStore + WriteStore + Clone + Send + Sync + 'static,
{
    type Result = ();

    async fn process_commit_archive(&self, archive_data: &CommitArchiveData) -> anyhow::Result<()> {
        // 1. Deserialize and verify the commit
        let commit: Commit = bcs::from_bytes(&archive_data.commit)
            .map_err(|e| ConsensusError::MalformedCommit(e))?;

        info!("Processing commit {} from archive", commit.index());

        // 2. Check if we already have this commit
        if let Some(existing) = self.store.get_commit_by_index(commit.index()) {
            debug!("Commit {} already exists, skipping", commit.index());
            return Ok(());
        }

        // 3. Verify previous commit exists (except for genesis)
        if commit.index() > 0 {
            let prev_commit = self
                .store
                .get_commit_by_index(commit.index() - 1)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Missing previous commit {} for commit {}",
                        commit.index() - 1,
                        commit.index()
                    )
                })?;

            if prev_commit.commit_ref.digest != commit.previous_digest() {
                return Err(anyhow::anyhow!(
                    "Previous digest mismatch for commit {}: expected {:?}, got {:?}",
                    commit.index(),
                    prev_commit.commit_ref.digest,
                    commit.previous_digest()
                ));
            }
        }

        // 4. Verify certifier blocks and accumulate votes
        let committee = self
            .store
            .get_committee(commit.epoch())?
            .ok_or_else(|| SomaError::NoCommitteeForEpoch(commit.epoch()))?;

        let mut stake_aggregator = StakeAggregator::<QuorumThreshold>::new();
        let commit_ref = types::consensus::commit::CommitRef {
            index: commit.index(),
            digest: TrustedCommit::compute_digest(&archive_data.commit),
        };

        let mut verified_certifier_blocks = Vec::new();
        for block_bytes in &archive_data.certifier_blocks {
            let signed_block: SignedBlock =
                bcs::from_bytes(block_bytes).map_err(ConsensusError::MalformedBlock)?;

            // Verify block signature
            self.block_verifier.verify(&signed_block)?;

            // Check if this block votes for our commit
            for vote in signed_block.commit_votes() {
                if *vote == commit_ref {
                    stake_aggregator.add(signed_block.author(), &committee);
                }
            }

            verified_certifier_blocks.push(VerifiedBlock::new_verified(
                signed_block,
                block_bytes.clone(),
            ));
        }

        // Verify we have quorum
        if !stake_aggregator.reached_threshold(&committee) {
            return Err(ConsensusError::NotEnoughCommitVotes {
                stake: stake_aggregator.stake(),
                peer: "archive".to_string(),
                commit: Box::new(commit.clone()),
            }
            .into());
        }

        // 5. Verify all commit blocks
        let mut verified_blocks = Vec::new();
        for block_bytes in &archive_data.blocks {
            let signed_block: SignedBlock =
                bcs::from_bytes(block_bytes).map_err(ConsensusError::MalformedBlock)?;

            self.block_verifier.verify(&signed_block)?;

            verified_blocks.push(VerifiedBlock::new_verified(
                signed_block,
                block_bytes.clone(),
            ));
        }

        // 6. Create TrustedCommit
        let trusted_commit = TrustedCommit::new_trusted(commit, archive_data.commit.clone());

        // 7. Write to ConsensusStore atomically (blocks + commit)
        // Include both commit blocks and certifier blocks
        let mut all_blocks = verified_blocks.clone();
        all_blocks.extend(verified_certifier_blocks);

        self.store.write(WriteBatch::new(
            all_blocks,
            vec![trusted_commit.clone()],
            vec![],
        ))?;

        // 8. Build CommittedSubDag and update WriteStore
        let sub_dag = CommittedSubDag::new(
            trusted_commit.leader(),
            verified_blocks,
            trusted_commit.timestamp_ms(),
            trusted_commit.reference(),
            trusted_commit.previous_digest(),
        );

        self.store.insert_commit(sub_dag.clone())?;
        self.store.update_highest_synced_commit(&sub_dag)?;

        info!(
            "Successfully synced commit {} from archive",
            trusted_commit.index()
        );

        Ok(())
    }
}
