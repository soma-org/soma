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

use super::encoder_core_thread::EncoderCoreThreadDispatcher;
use crate::types::signed::Signed;

pub(crate) struct EncoderService<C: EncoderCoreThreadDispatcher, S: Store> {
    context: Arc<EncoderContext>,
    core_dispatcher: Arc<C>,
    store: Arc<S>,
    keypair: Arc<ProtocolKeyPair>,
}

impl<C: EncoderCoreThreadDispatcher, S: Store> EncoderService<C, S> {
    pub(crate) fn new(
        context: Arc<EncoderContext>,
        core_dispatcher: Arc<C>,
        store: Arc<S>,
        keypair: Arc<ProtocolKeyPair>,
    ) -> Self {
        println!("configured core thread");
        Self {
            context,
            core_dispatcher,
            store,
            keypair,
        }
    }
}

fn unverified<T>(input: &T) -> ShardResult<()> {
    Ok(())
}

#[async_trait]
impl<C: EncoderCoreThreadDispatcher, S: Store> EncoderNetworkService for EncoderService<C, S> {
    async fn handle_send_shard_input(
        &self,
        peer: NetworkingIndex,
        shard_input_bytes: Bytes,
    ) -> ShardResult<()> {
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
        let shard = self.store.read_shard(&shard_commit.shard_ref())?;
        if !shard.contains(&peer) {
            return Err(ShardError::UnauthorizedPeer);
        }
        //2. verify signature

        shard_commit.verify_signature(
            crate::Scope::ShardCommit,
            &self.context.network_committee.identity(&peer).protocol_key,
        )?;

        //3. verify shard commit
        //TODO: fix verification
        let verified_shard_commit = Verified::new(shard_commit, shard_commit_bytes, unverified)?;

        if self
            .store
            .read_shard_commit_digest(&verified_shard_commit.shard_ref(), peer)
            .is_ok_and(|digest| digest != verified_shard_commit.digest())
        {
            return Err(ShardError::ConflictingRequest);
        }

        // TODO: add digest to datastore

        let signature_bytes = bcs::to_bytes(
            &Signed::new(
                verified_shard_commit.deref().to_owned(),
                crate::Scope::ShardReveal,
                &self.keypair,
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
        unimplemented!()
    }

    async fn handle_batch_get_shard_commit_certificates(
        &self,
        peer: NetworkingIndex,
        slots_bytes: Bytes,
    ) -> ShardResult<Vec<Serialized<ShardCertificate<Signed<ShardCommit>>>>> {
        let shard_slots: ShardSlots =
            bcs::from_bytes(&slots_bytes).map_err(ShardError::MalformedType)?;
        let shard = self.store.read_shard(&shard_slots.shard_ref())?;
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

    async fn handle_get_shard_reveal_signature(
        &self,
        peer: NetworkingIndex,
        shard_reveal_bytes: Bytes,
    ) -> ShardResult<Serialized<Signature<Signed<ShardReveal>>>> {
        let shard_reveal: Signed<ShardReveal> =
            bcs::from_bytes(&shard_reveal_bytes).map_err(ShardError::MalformedType)?;

        //1. verify shard membersip
        let shard = self.store.read_shard(&shard_reveal.shard_ref())?;
        if !shard.contains(&peer) {
            return Err(ShardError::UnauthorizedPeer);
        }
        //2. verify signature

        shard_reveal.verify_signature(
            crate::Scope::ShardReveal,
            &self.context.network_committee.identity(&peer).protocol_key,
        )?;

        //3. verify shard reveal
        //TODO: fix verification
        let verified_shard_reveal = Verified::new(shard_reveal, shard_reveal_bytes, unverified)?;

        //4. check for digest in store
        if self
            .store
            .read_shard_reveal_digest(&verified_shard_reveal.shard_ref(), peer)
            .is_ok_and(|digest| digest != verified_shard_reveal.digest())
        {
            return Err(ShardError::ConflictingRequest);
        }

        // TODO: add digest to datastore
        // TODO: clean up how signatures are created

        let signature_bytes = bcs::to_bytes(
            &Signed::new(
                verified_shard_reveal.deref().to_owned(),
                crate::Scope::ShardReveal,
                &self.keypair,
            )?
            .signature(),
        )
        .map_err(ShardError::SerializationFailure)?;
        Ok(Serialized::new(bytes::Bytes::copy_from_slice(
            &signature_bytes,
        )))
    }

    async fn handle_send_shard_reveal_certificate(
        &self,
        peer: NetworkingIndex,
        shard_reveal_certificate_bytes: Bytes,
    ) -> ShardResult<()> {
        unimplemented!()
    }

    async fn handle_batch_get_shard_reveal_certificates(
        &self,
        peer: NetworkingIndex,
        slots_bytes: Bytes,
    ) -> ShardResult<Vec<Serialized<ShardCertificate<Signed<ShardReveal>>>>> {
        let shard_slots: ShardSlots =
            bcs::from_bytes(&slots_bytes).map_err(ShardError::MalformedType)?;
        let shard = self.store.read_shard(&shard_slots.shard_ref())?;
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
        unimplemented!()
    }

    async fn handle_batch_send_shard_removal_certificates(
        &self,
        peer: NetworkingIndex,
        shard_removal_certificates: Vec<Bytes>,
    ) -> ShardResult<()> {
        unimplemented!()
    }

    async fn handle_send_shard_endorsement(
        &self,
        peer: NetworkingIndex,
        shard_endorsement_bytes: Bytes,
    ) -> ShardResult<()> {
        unimplemented!()
    }

    async fn handle_send_shard_finality_proof(
        &self,
        peer: NetworkingIndex,
        shard_finality_proof_bytes: Bytes,
    ) -> ShardResult<()> {
        unimplemented!()
    }

    async fn handle_send_shard_delivery_proof(
        &self,
        peer: NetworkingIndex,
        shard_delivery_proof_bytes: Bytes,
    ) -> ShardResult<()> {
        unimplemented!()
    }
}
