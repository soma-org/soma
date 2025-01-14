use async_trait::async_trait;
use bytes::Bytes;
use std::ops::Deref;
use std::sync::Arc;

use crate::{
    error::{ShardError, ShardResult},
    networking::messaging::EncoderNetworkService,
    storage::datastore::Store,
    types::{
        certificate::ShardCertificate,
        context::EncoderContext,
        network_committee::NetworkingIndex,
        serialized::Serialized,
        shard::ShardRef,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_input::ShardInput,
        shard_reveal::{ShardReveal, ShardRevealAPI},
        shard_slots::{ShardSlots, ShardSlotsAPI},
        signed::Signature,
        verified::Verified,
    },
    ProtocolKeyPair,
};

use crate::types::signed::Signed;

use super::pipeline_dispatcher::PipelineDispatcher;

pub(crate) struct EncoderService<PD: PipelineDispatcher, S: Store> {
    context: Arc<EncoderContext>,
    pipeline_dispatcher: Arc<PD>, //TODO: confirm this needs an arc?
    store: Arc<S>,
    protocol_keypair: Arc<ProtocolKeyPair>,
}

impl<PD: PipelineDispatcher, S: Store> EncoderService<PD, S> {
    pub(crate) fn new(
        context: Arc<EncoderContext>,
        pipeline_dispatcher: Arc<PD>,
        store: Arc<S>,
        protocol_keypair: Arc<ProtocolKeyPair>,
    ) -> Self {
        println!("configured core thread");
        Self {
            context,
            pipeline_dispatcher,
            store,
            protocol_keypair,
        }
    }
}

fn unverified<T>(input: &T) -> ShardResult<()> {
    Ok(())
}

#[async_trait]
impl<PD: PipelineDispatcher, S: Store> EncoderNetworkService for EncoderService<PD, S> {
    async fn handle_send_shard_input(
        &self,
        peer: NetworkingIndex,
        shard_input_bytes: Bytes,
    ) -> ShardResult<()> {
        // 1. verify everything in the input (tx, amount, finality, shard inclusion)
        // 2. store shard members in shard store
        // 3. send to the task manager

        // inside task manager:
        // 1. download the data
        // 2. compute the embeddings for the data
        // 3. broadcast the commit out to the rest of the shard
        unimplemented!()
    }
    async fn handle_get_shard_input(
        &self,
        peer: NetworkingIndex,
        shard_ref_bytes: Bytes,
    ) -> ShardResult<Serialized<Signed<ShardInput>>> {
        let shard_ref: ShardRef =
            bcs::from_bytes(&shard_ref_bytes).map_err(ShardError::MalformedType)?;
        // TODO: change this to use a cache rather than database call?
        let shard = self.store.read_shard(&shard_ref)?;
        if !shard.contains(&peer) {
            return Err(ShardError::UnauthorizedPeer);
        }
        let signed_shard_input = self.store.read_signed_shard_input(&shard_ref)?;
        Ok(signed_shard_input.serialized())
    }
    async fn handle_get_shard_commit_signature(
        &self,
        peer: NetworkingIndex,
        shard_commit_bytes: Bytes,
    ) -> ShardResult<Serialized<Signature<Signed<ShardCommit>>>> {
        let shard_commit: Signed<ShardCommit> =
            bcs::from_bytes(&shard_commit_bytes).map_err(ShardError::MalformedType)?;

        //1. verify shard membersip
        let shard = self.store.read_shard(shard_commit.shard_ref())?;
        if !shard.contains(&peer) {
            return Err(ShardError::UnauthorizedPeer);
        }
        //2. verify signature

        shard_commit.verify_signature(
            crate::Scope::ShardCommit,
            &self.context.network_committee.identity(peer).protocol_key,
        )?;

        //3. verify shard commit
        //TODO: fix verification
        let verified_shard_commit = Verified::new(shard_commit, shard_commit_bytes, unverified)?;

        if self
            .store
            .read_shard_commit_digest(verified_shard_commit.shard_ref(), peer)
            .is_ok_and(|digest| digest != verified_shard_commit.digest())
        {
            return Err(ShardError::ConflictingRequest);
        }

        // TODO: add digest to datastore

        let signature_bytes = bcs::to_bytes(
            &Signed::new(
                verified_shard_commit.deref().to_owned(),
                crate::Scope::ShardReveal,
                &self.protocol_keypair,
            )?
            .signature(),
        )
        .map_err(ShardError::SerializationFailure)?;
        Ok(Serialized::new(bytes::Bytes::copy_from_slice(
            &signature_bytes,
        )))
    }

    async fn handle_send_shard_commit_certificate(
        &self,
        peer: NetworkingIndex,
        shard_commit_certificate_bytes: Bytes,
    ) -> ShardResult<()> {
        // 1. verify the shard commit certificate
        // 2. verify the shard commit values
        // 3. send to task manager
        // task manager:
        // 1. trigger downloading the commit
        // (downloading includes encrypted data and probe)
        // 2. add to tracker for the quorum
        // once quorum has hit, trigger the timeouts for checking with peers and sending removal signatures.
        // also need to figure out the right way to deal with downloading data too
        // !!!!!!! MAKE IT SO A COMMIT IS DONE WHEN THE DATA IS DOWNLOADED
        // after the timeout hits, ask peers for commits but also ask for data
        unimplemented!()
    }

    async fn handle_batch_get_shard_commit_certificates(
        &self,
        peer: NetworkingIndex,
        slots_bytes: Bytes,
    ) -> ShardResult<Vec<Serialized<ShardCertificate<Signed<ShardCommit>>>>> {
        let shard_slots: ShardSlots =
            bcs::from_bytes(&slots_bytes).map_err(ShardError::MalformedType)?;
        let shard = self.store.read_shard(shard_slots.shard_ref())?;
        if !shard.contains(&peer) {
            return Err(ShardError::UnauthorizedPeer);
        }

        let batch_shard_commit_certificates = self.store.batch_read_shard_commit_certificates(
            shard_slots.shard_ref().clone(),
            shard_slots.shard_members(),
        )?;

        Ok(batch_shard_commit_certificates
            .iter()
            .flatten()
            .map(|x| x.serialized())
            .collect())
    }

    // async fn handle_get_shard_reveal_signature(
    //     &self,
    //     peer: NetworkingIndex,
    //     shard_reveal_bytes: Bytes,
    // ) -> ShardResult<Serialized<Signature<Signed<ShardReveal>>>> {
    //     let shard_reveal: Signed<ShardReveal> =
    //         bcs::from_bytes(&shard_reveal_bytes).map_err(ShardError::MalformedType)?;

    //     //1. verify shard membersip
    //     let shard = self.store.read_shard(shard_reveal.shard_ref())?;
    //     if !shard.contains(&peer) {
    //         return Err(ShardError::UnauthorizedPeer);
    //     }
    //     //2. verify signature

    //     shard_reveal.verify_signature(
    //         crate::Scope::ShardReveal,
    //         &self.context.network_committee.identity(peer).protocol_key,
    //     )?;

    //     //3. verify shard reveal
    //     //TODO: fix verification
    //     let verified_shard_reveal = Verified::new(shard_reveal, shard_reveal_bytes, unverified)?;

    //     //4. check for digest in store
    //     if self
    //         .store
    //         .read_shard_reveal_digest(verified_shard_reveal.shard_ref(), peer)
    //         .is_ok_and(|digest| digest != verified_shard_reveal.digest())
    //     {
    //         return Err(ShardError::ConflictingRequest);
    //     }

    //     // TODO: add digest to datastore
    //     // TODO: clean up how signatures are created

    //     let signature_bytes = bcs::to_bytes(
    //         &Signed::new(
    //             verified_shard_reveal.deref().to_owned(),
    //             crate::Scope::ShardReveal,
    //             &self.protocol_keypair,
    //         )?
    //         .signature(),
    //     )
    //     .map_err(ShardError::SerializationFailure)?;
    //     Ok(Serialized::new(bytes::Bytes::copy_from_slice(
    //         &signature_bytes,
    //     )))
    // }

    async fn handle_send_shard_reveal_certificate(
        &self,
        peer: NetworkingIndex,
        shard_reveal_certificate_bytes: Bytes,
    ) -> ShardResult<()> {
        // 1. verify the shard reveal certificate
        // 2. verify the shard reveal values
        // 3. send to task manager
        // task manager:
        // 1. decrypt data
        // 2. apply probe
        // 3. store partial result
        // wait for quorum to be done and then trigger timeout
        // done status is after the computation has been completed for that probe combo
        unimplemented!()
    }

    async fn handle_batch_get_shard_reveal_certificates(
        &self,
        peer: NetworkingIndex,
        slots_bytes: Bytes,
    ) -> ShardResult<Vec<Serialized<ShardCertificate<Signed<ShardReveal>>>>> {
        let shard_slots: ShardSlots =
            bcs::from_bytes(&slots_bytes).map_err(ShardError::MalformedType)?;
        let shard = self.store.read_shard(shard_slots.shard_ref())?;
        if !shard.contains(&peer) {
            return Err(ShardError::UnauthorizedPeer);
        }

        let batch_shard_reveal_certificates = self.store.batch_read_shard_reveal_certificates(
            shard_slots.shard_ref().clone(),
            shard_slots.shard_members(),
        )?;

        Ok(batch_shard_reveal_certificates
            .iter()
            .flatten()
            .map(|x| x.serialized())
            .collect())
    }

    async fn handle_batch_send_shard_removal_signatures(
        &self,
        peer: NetworkingIndex,
        shard_removal_signatures: Vec<Bytes>,
    ) -> ShardResult<()> {
        // verify that removal shard signature is valid
        // check whether the software already has a certified removal
        // send to task manager
        // task manager:
        // stop tasks pertaining to that peer
        unimplemented!()
    }

    async fn handle_batch_send_shard_removal_certificates(
        &self,
        peer: NetworkingIndex,
        shard_removal_certificates: Vec<Bytes>,
    ) -> ShardResult<()> {
        unimplemented!()
        // verify validity of certificate
        // if the certificate is new:
        // 1. trigger removal and stopping of tokio tasks related to this peer
        // otherwise store the entire set if it is a superset or the same certificates as your own node
        // broadcast a message with your new super set to all peers
    }

    async fn handle_send_shard_endorsement(
        &self,
        peer: NetworkingIndex,
        shard_endorsement_bytes: Bytes,
    ) -> ShardResult<()> {
        // collect enough for quorum endorsement
        // trigger staggered timeout to submit on-chain and back to the RPC
        unimplemented!()
    }

    async fn handle_send_shard_completion_proof(
        &self,
        peer: NetworkingIndex,
        shard_completion_proof_bytes: Bytes,
    ) -> ShardResult<()> {
        // check validity of proof
        // send to task manager to trigger stopping the countdown for delivery and finality
        // handle clean up
        unimplemented!()
    }
}
