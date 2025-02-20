use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::{bls12381::min_sig, traits::KeyPair, traits::Signer};
use shared::{
    crypto::keys::{EncoderKeyPair, EncoderPublicKey, ProtocolKeyPair},
    digest::Digest,
    metadata::verify_metadata,
    scope::Scope,
    serialized::Serialized,
    signed::{Signature, Signed},
    verified::Verified,
};
use std::sync::Arc;

use crate::{
    actors::{workers::vdf::VDFProcessor, ActorHandle},
    error::{ShardError, ShardResult},
    networking::messaging::EncoderInternalNetworkService,
    storage::datastore::Store,
    types::{
        certified::{Certified, CertifiedAPI},
        encoder_committee::EncoderIndex,
        encoder_context::EncoderContext,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_reveal::{ShardReveal, ShardRevealAPI},
        shard_verifier::ShardVerifier,
        shard_votes::{CommitRound, RevealRound, ShardVotes, ShardVotesAPI},
    },
};

use super::pipeline_dispatcher::PipelineDispatcher;

pub(crate) struct EncoderInternalService<PD: PipelineDispatcher> {
    context: Arc<EncoderContext>,
    pipeline_dispatcher: Arc<PD>, //TODO: confirm this needs an arc?
    vdf: ActorHandle<VDFProcessor>,
    shard_verifier: ShardVerifier,
    store: Arc<dyn Store>,
    encoder_keypair: Arc<EncoderKeyPair>,
}

impl<PD: PipelineDispatcher> EncoderInternalService<PD> {
    pub(crate) fn new(
        context: Arc<EncoderContext>,
        pipeline_dispatcher: Arc<PD>,
        vdf: ActorHandle<VDFProcessor>,
        shard_verifier: ShardVerifier,
        store: Arc<dyn Store>,
        encoder_keypair: Arc<EncoderKeyPair>,
    ) -> Self {
        println!("configured core thread");
        Self {
            context,
            pipeline_dispatcher,
            vdf,
            shard_verifier,
            store,
            encoder_keypair,
        }
    }
}

#[async_trait]
impl<PD: PipelineDispatcher> EncoderInternalNetworkService for EncoderInternalService<PD> {
    async fn handle_send_commit(
        &self,
        peer: EncoderIndex,
        commit_bytes: Bytes,
    ) -> ShardResult<
        Serialized<
            Signature<Signed<ShardCommit, min_sig::BLS12381Signature>, min_sig::BLS12381Signature>,
        >,
    > {
        // convert into correct type
        let signed_commit: Signed<ShardCommit, min_sig::BLS12381Signature> =
            bcs::from_bytes(&commit_bytes).map_err(ShardError::MalformedType)?;
        let (auth_token_digest, shard) = self
            .shard_verifier
            .verify(&self.context, &self.vdf, signed_commit.auth_token())
            .await?;

        // perform verification on type and auth including signature checks
        let verified_commit = Verified::new(signed_commit.clone(), commit_bytes, |signed_commit| {
            // check slot is valid member of computational set
            if !shard.inference_set().contains(&signed_commit.slot()) {
                return Err(shared::error::SharedError::ValidationError("s".to_string()));
            }
            if let Some(signed_route) = signed_commit.route() {
                // check route destination is valid (someone who is not a member of the computation set) and not the slot
                if shard.inference_set().contains(&signed_route.destination())
                    || &signed_route.destination() == &signed_commit.slot()
                {
                    return Err(shared::error::SharedError::ValidationError("s".to_string()));
                }

                // digest of route matches the auth token
                if signed_route.auth_token_digest() != auth_token_digest {
                    return Err(shared::error::SharedError::ValidationError("s".to_string()));
                }

                // check signature of route is by the slot
                let _ = signed_route.verify(
                    Scope::ShardCommitRoute,
                    self.context
                        .encoder_committee
                        .encoder(signed_commit.slot())
                        .encoder_key
                        .inner(),
                )?;
            }

            // check overall signature is by the committer (slot or route destination if it exists)
            let _ = signed_commit.verify(
                Scope::ShardCommitRoute,
                self.context
                    .encoder_committee
                    .encoder(signed_commit.committer())
                    .encoder_key
                    .inner(),
            )?;

            let metadata = signed_commit.commit();
            // TODO: update to actually check commit size, shape, etc according to embedding standards
            let _ = verify_metadata(None, None, None, None, None)(metadata)?;
            Ok(())
        })
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

        self.store.atomic_commit(
            Digest::new(&shard).map_err(ShardError::DigestFailure)?,
            signed_commit.clone(),
        )?;

        let keypair = self.encoder_keypair.inner().copy();

        let partial_sig = Signed::new(signed_commit, Scope::ShardCertificate, &keypair.private())
            .map_err(ShardError::SerializationFailure)?;

        Ok(partial_sig.serialized())

        // issue signature if there are no conflicts
    }
    async fn handle_send_certified_commit(
        &self,
        peer: EncoderIndex,
        certified_commit_bytes: Bytes,
    ) -> ShardResult<()> {
        // convert into correct type
        let certified_commit: Certified<Signed<ShardCommit, min_sig::BLS12381Signature>> =
            bcs::from_bytes(&certified_commit_bytes).map_err(ShardError::MalformedType)?;
        let (auth_token_digest, shard) = self
            .shard_verifier
            .verify(&self.context, &self.vdf, certified_commit.auth_token())
            .await?;
        // perform verification on type and auth including signature checks
        let verified_certified_commit = Verified::new(
            certified_commit,
            certified_commit_bytes,
            |certified_commit| {
                let certifier_indices = certified_commit.indices();

                for index in &certifier_indices {
                    if !shard.evaluation_set().contains(index) {
                        return Err(shared::error::SharedError::ValidationError(
                            "index not in evaluation set".to_string(),
                        ));
                    }
                }

                if certifier_indices.len() < shard.evaluation_quorum_threshold() as usize {
                    return Err(shared::error::SharedError::ValidationError(format!(
                        "got: {:?}, needed: {}",
                        certifier_indices,
                        shard.evaluation_quorum_threshold()
                    )));
                }

                certified_commit
                    .verify(Scope::ShardCertificate, &self.context.encoder_committee)?;
                Ok(())
            },
        )
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;
        // TODO: send to orchestrator
        Ok(())
    }
    async fn handle_send_commit_votes(
        &self,
        peer: EncoderIndex,
        votes_bytes: Bytes,
    ) -> ShardResult<()> {
        // convert into correct type
        let votes: Signed<ShardVotes<CommitRound>, min_sig::BLS12381Signature> =
            bcs::from_bytes(&votes_bytes).map_err(ShardError::MalformedType)?;
        let (auth_token_digest, shard) = self
            .shard_verifier
            .verify(&self.context, &self.vdf, votes.auth_token())
            .await?;

        let verified_commit_votes = Verified::new(votes, votes_bytes, |votes| {
            // verify that the author is part of the evaluation set
            if !shard.evaluation_set().contains(&votes.voter()) {
                return Err(shared::error::SharedError::ValidationError(
                    "voter is not in evaluation set".to_string(),
                ));
            }
            // verify that the reject encoder indexs are valid slots
            for index in votes.rejects() {
                if !shard.inference_set().contains(index) {
                    return Err(shared::error::SharedError::ValidationError(
                        "index not in inference set".to_string(),
                    ));
                }
            }
            // verify signature matches the voter
            let _ = votes.verify(
                Scope::ShardCommitVotes,
                self.context
                    .encoder_committee
                    .encoder(votes.voter())
                    .encoder_key
                    .inner(),
            )?;

            Ok(())
        })
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;
        // send to orchestrator
        Ok(())
    }
    async fn handle_send_reveal(&self, peer: EncoderIndex, reveal_bytes: Bytes) -> ShardResult<()> {
        // convert into correct type
        let reveal: Signed<ShardReveal, min_sig::BLS12381Signature> =
            bcs::from_bytes(&reveal_bytes).map_err(ShardError::MalformedType)?;
        let (auth_token_digest, shard) = self
            .shard_verifier
            .verify(&self.context, &self.vdf, reveal.auth_token())
            .await?;
        // perform verification on type and auth including signature checks
        // verify that the signature matches the slot and that the slot is valid for the shard
        // database lookup for the commit metadata and verify the encryption key
        let verified_commit = Verified::new(reveal, reveal_bytes, |r| Ok(()))
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

        Ok(())
    }
    async fn handle_send_reveal_votes(
        &self,
        peer: EncoderIndex,
        votes_bytes: Bytes,
    ) -> ShardResult<()> {
        // convert into correct type
        let votes: Signed<ShardVotes<RevealRound>, min_sig::BLS12381Signature> =
            bcs::from_bytes(&votes_bytes).map_err(ShardError::MalformedType)?;
        let (auth_token_digest, shard) = self
            .shard_verifier
            .verify(&self.context, &self.vdf, votes.auth_token())
            .await?;
        // perform verification on type and auth including signature checks
        // verify signature matches the author
        // verify that the author is part of the evaluation set
        // verify that the reject encoder indexs are valid slots
        let verified_reveal_votes = Verified::new(votes, votes_bytes, |votes| {
            // verify that the author is part of the evaluation set
            if !shard.evaluation_set().contains(&votes.voter()) {
                return Err(shared::error::SharedError::ValidationError(
                    "voter is not in evaluation set".to_string(),
                ));
            }
            // verify that the reject encoder indexs are valid slots
            for index in votes.rejects() {
                if !shard.inference_set().contains(index) {
                    return Err(shared::error::SharedError::ValidationError(
                        "index not in inference set".to_string(),
                    ));
                }
            }
            // verify signature matches the voter
            let _ = votes.verify(
                Scope::ShardRevealVotes,
                self.context
                    .encoder_committee
                    .encoder(votes.voter())
                    .encoder_key
                    .inner(),
            )?;

            Ok(())
        })
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;
        // send to orchestrator
        Ok(())
    }
}
