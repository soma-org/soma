use async_trait::async_trait;
use bytes::Bytes;
use std::sync::Arc;

use crate::{
    error::{ShardError, ShardResult},
    networking::messaging::EncoderNetworkService,
    types::{
        certificate::ShardCertificate, network_committee::NetworkingIndex, serialized::Serialized,
        shard::ShardRef, shard_commit::ShardCommit, shard_input::ShardInput,
        shard_reveal::ShardReveal, signed::Signature, verified::Verified,
    },
};

use super::encoder_core_thread::EncoderCoreThreadDispatcher;
use crate::types::signed::Signed;

pub(crate) struct EncoderService<C: EncoderCoreThreadDispatcher> {
    core_dispatcher: Arc<C>,
}

impl<C: EncoderCoreThreadDispatcher> EncoderService<C> {
    pub(crate) fn new(core_dispatcher: Arc<C>) -> Self {
        println!("configured core thread");
        Self { core_dispatcher }
    }
}

fn unverified<T>(input: &T) -> ShardResult<()> {
    Ok(())
}

#[async_trait]
impl<C: EncoderCoreThreadDispatcher> EncoderNetworkService for EncoderService<C> {
    async fn handle_send_shard_input(
        &self,
        peer: NetworkingIndex,
        shard_input_bytes: Bytes,
    ) -> ShardResult<()> {
        let signed_input: Signed<ShardInput> =
            bcs::from_bytes(&shard_input_bytes).map_err(ShardError::MalformedType)?;
        let verified_input = Verified::new(signed_input, shard_input_bytes, &unverified)?;
        self.core_dispatcher
            .process_shard_input(verified_input)
            .await?;
        Ok(())
    }
    async fn handle_get_shard_input(
        &self,
        peer: NetworkingIndex,
        shard_ref_bytes: Bytes,
    ) -> ShardResult<Serialized<Signed<ShardInput>>> {
        let shard_ref: ShardRef =
            bcs::from_bytes(&shard_ref_bytes).map_err(ShardError::MalformedType)?;
        let verified_shard_ref = Verified::new(shard_ref, shard_ref_bytes, &unverified)?;
        // self.core_dispatcher.process_input(verified_input).await?;
        unimplemented!()
    }
    async fn handle_get_shard_commit_signature(
        &self,
        peer: NetworkingIndex,
        shard_commit_bytes: Bytes,
    ) -> ShardResult<Serialized<Signature<Signed<ShardCommit>>>> {
        let shard_commit: Signed<ShardCommit> =
            bcs::from_bytes(&shard_commit_bytes).map_err(ShardError::MalformedType)?;
        let verified_shard_commit = Verified::new(shard_commit, shard_commit_bytes, &unverified)?;
        // self.core_dispatcher.process_input(verified_input).await?;

        unimplemented!()
    }

    async fn handle_send_shard_commit_certificate(
        &self,
        peer: NetworkingIndex,
        shard_commit_certificate_bytes: Bytes,
    ) -> ShardResult<()> {
        let shard_commit_certificate: ShardCertificate<Signed<ShardCommit>> =
            bcs::from_bytes(&shard_commit_certificate_bytes).map_err(ShardError::MalformedType)?;
        let verified_shard_commit_certificate = Verified::new(
            shard_commit_certificate,
            shard_commit_certificate_bytes,
            &unverified,
        )?;
        // self.core_dispatcher.process_input(verified_input).await?;
        Ok(())
    }

    async fn handle_batch_get_shard_commit_certificates(
        &self,
        peer: NetworkingIndex,
        slots_bytes: Bytes,
    ) -> ShardResult<Vec<Serialized<ShardCertificate<Signed<ShardCommit>>>>> {
        unimplemented!()
    }

    async fn handle_get_shard_reveal_signature(
        &self,
        peer: NetworkingIndex,
        shard_reveal_bytes: Bytes,
    ) -> ShardResult<Serialized<Signature<Signed<ShardReveal>>>> {
        unimplemented!()
    }

    async fn handle_send_shard_reveal_certificate(
        &self,
        peer: NetworkingIndex,
        shard_reveal_certificate_bytes: Bytes,
    ) -> ShardResult<()> {
        Ok(())
    }

    async fn handle_batch_get_shard_reveal_certificates(
        &self,
        peer: NetworkingIndex,
        slots_bytes: Bytes,
    ) -> ShardResult<Vec<Serialized<ShardCertificate<Signed<ShardReveal>>>>> {
        // Ok(())
        unimplemented!()
    }

    async fn handle_batch_send_shard_removal_signatures(
        &self,
        peer: NetworkingIndex,
        shard_removal_signatures: Vec<Bytes>,
    ) -> ShardResult<()> {
        Ok(())
    }

    async fn handle_batch_send_shard_removal_certificates(
        &self,
        peer: NetworkingIndex,
        shard_removal_certificates: Vec<Bytes>,
    ) -> ShardResult<()> {
        Ok(())
    }

    async fn handle_send_shard_endorsement(
        &self,
        peer: NetworkingIndex,
        shard_endorsement_bytes: Bytes,
    ) -> ShardResult<()> {
        Ok(())
    }

    async fn handle_send_shard_finality_proof(
        &self,
        peer: NetworkingIndex,
        shard_finality_proof_bytes: Bytes,
    ) -> ShardResult<()> {
        Ok(())
    }

    async fn handle_send_shard_delivery_proof(
        &self,
        peer: NetworkingIndex,
        shard_delivery_proof_bytes: Bytes,
    ) -> ShardResult<()> {
        Ok(())
    }
}
