use crate::authority::StableSyncAuthoritySigner;
use crate::authority_per_epoch_store::AuthorityPerEpochStore;
use crate::checkpoints::CheckpointStore;
use crate::consensus_adapter::SubmitToConsensus;
use crate::reconfiguration::ReconfigurationInitiator;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info, instrument, trace};
use types::base::AuthorityName;
use types::checkpoints::{
    CertifiedCheckpointSummary, CheckpointContents, CheckpointSignatureMessage, CheckpointSummary,
    SignedCheckpointSummary, VerifiedCheckpoint,
};
use types::consensus::ConsensusTransaction;
use types::envelope::Message as _;
use types::error::SomaResult;

#[async_trait]
pub trait CheckpointOutput: Sync + Send + 'static {
    async fn checkpoint_created(
        &self,
        summary: &CheckpointSummary,
        contents: &CheckpointContents,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        checkpoint_store: &Arc<CheckpointStore>,
    ) -> SomaResult;
}

#[async_trait]
pub trait CertifiedCheckpointOutput: Sync + Send + 'static {
    async fn certified_checkpoint_created(
        &self,
        summary: &CertifiedCheckpointSummary,
    ) -> SomaResult;
}

pub struct SubmitCheckpointToConsensus<T> {
    pub sender: T,
    pub signer: StableSyncAuthoritySigner,
    pub authority: AuthorityName,
    pub next_reconfiguration_timestamp_ms: u64,
}

pub struct LogCheckpointOutput;

impl LogCheckpointOutput {
    pub fn boxed() -> Box<dyn CheckpointOutput> {
        Box::new(Self)
    }

    pub fn boxed_certified() -> Box<dyn CertifiedCheckpointOutput> {
        Box::new(Self)
    }
}

#[async_trait]
impl<T: SubmitToConsensus + ReconfigurationInitiator> CheckpointOutput
    for SubmitCheckpointToConsensus<T>
{
    #[instrument(level = "debug", skip_all)]
    async fn checkpoint_created(
        &self,
        summary: &CheckpointSummary,
        contents: &CheckpointContents,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        checkpoint_store: &Arc<CheckpointStore>,
    ) -> SomaResult {
        LogCheckpointOutput
            .checkpoint_created(summary, contents, epoch_store, checkpoint_store)
            .await?;

        let checkpoint_timestamp = summary.timestamp_ms;
        let checkpoint_seq = summary.sequence_number;

        let highest_verified_checkpoint =
            checkpoint_store.get_highest_verified_checkpoint()?.map(|x| *x.sequence_number());

        if Some(checkpoint_seq) > highest_verified_checkpoint {
            debug!(
                "Sending checkpoint signature at sequence {checkpoint_seq} to consensus, timestamp {checkpoint_timestamp}.
                {}ms left till end of epoch at timestamp {}",
                self.next_reconfiguration_timestamp_ms.saturating_sub(checkpoint_timestamp), self.next_reconfiguration_timestamp_ms
            );

            let summary = SignedCheckpointSummary::new(
                epoch_store.epoch(),
                summary.clone(),
                &*self.signer,
                self.authority,
            );

            let message = CheckpointSignatureMessage { summary };
            let transaction = ConsensusTransaction::new_checkpoint_signature_message(message);
            self.sender.submit_to_consensus(&vec![transaction], epoch_store)?;
        } else {
            debug!(
                "Checkpoint at sequence {checkpoint_seq} is already certified, skipping signature submission to consensus",
            );
        }

        if checkpoint_timestamp >= self.next_reconfiguration_timestamp_ms {
            // close_epoch is ok if called multiple times
            info!(
                "Closing epoch at sequence {checkpoint_seq} at timestamp {checkpoint_timestamp}. next_reconfiguration_timestamp_ms {}",
                self.next_reconfiguration_timestamp_ms
            );
            self.sender.close_epoch(epoch_store);
        }
        Ok(())
    }
}

#[async_trait]
impl CheckpointOutput for LogCheckpointOutput {
    async fn checkpoint_created(
        &self,
        summary: &CheckpointSummary,
        contents: &CheckpointContents,
        _epoch_store: &Arc<AuthorityPerEpochStore>,
        _checkpoint_store: &Arc<CheckpointStore>,
    ) -> SomaResult {
        trace!(
            "Including following transactions in checkpoint {}: {:?}",
            summary.sequence_number, contents
        );
        info!(
            "Creating checkpoint {:?} at epoch {}, sequence {}, previous digest {:?}, transactions count {}, content digest {:?}, end_of_epoch_data {:?}",
            summary.digest(),
            summary.epoch,
            summary.sequence_number,
            summary.previous_digest,
            contents.size(),
            summary.content_digest,
            summary.end_of_epoch_data,
        );

        Ok(())
    }
}

#[async_trait]
impl CertifiedCheckpointOutput for LogCheckpointOutput {
    async fn certified_checkpoint_created(
        &self,
        summary: &CertifiedCheckpointSummary,
    ) -> SomaResult {
        info!(
            "Certified checkpoint with sequence {} and digest {}",
            summary.sequence_number,
            summary.digest()
        );
        Ok(())
    }
}

pub struct SendCheckpointToStateSync {
    handle: sync::builder::StateSyncHandle,
}

impl SendCheckpointToStateSync {
    pub fn new(handle: sync::builder::StateSyncHandle) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl CertifiedCheckpointOutput for SendCheckpointToStateSync {
    #[instrument(level = "debug", skip_all)]
    async fn certified_checkpoint_created(
        &self,
        summary: &CertifiedCheckpointSummary,
    ) -> SomaResult {
        info!(
            "Certified checkpoint with sequence {} and digest {}",
            summary.sequence_number,
            summary.digest()
        );
        self.handle.send_checkpoint(VerifiedCheckpoint::new_unchecked(summary.to_owned())).await;

        Ok(())
    }
}
