use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::{bls12381::min_sig, traits::KeyPair, traits::Signer};
use shared::{
    crypto::keys::{EncoderKeyPair, EncoderPublicKey, ProtocolKeyPair},
    digest::Digest,
    error::SharedError,
    metadata::{verify_metadata, EncryptionAPI, MetadataAPI},
    scope::Scope,
    serialized::Serialized,
    signed::{Signature, Signed},
    verified::Verified,
};
use std::sync::Arc;

use crate::{
    actors::{pipelines::certified_commit, workers::vdf::VDFProcessor, ActorHandle},
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

use super::pipeline_dispatcher::Dispatcher;

pub(crate) struct EncoderInternalService<D: Dispatcher> {
    context: Arc<EncoderContext>,
    dispatcher: D,
    vdf: ActorHandle<VDFProcessor>,
    shard_verifier: ShardVerifier,
    store: Arc<dyn Store>,
    encoder_keypair: Arc<EncoderKeyPair>,
}

impl<D: Dispatcher> EncoderInternalService<D> {
    pub(crate) fn new(
        context: Arc<EncoderContext>,
        dispatcher: D,
        vdf: ActorHandle<VDFProcessor>,
        shard_verifier: ShardVerifier,
        store: Arc<dyn Store>,
        encoder_keypair: Arc<EncoderKeyPair>,
    ) -> Self {
        println!("configured core thread");
        Self {
            context,
            dispatcher,
            vdf,
            shard_verifier,
            store,
            encoder_keypair,
        }
    }
}

#[async_trait]
impl<D: Dispatcher> EncoderInternalNetworkService for EncoderInternalService<D> {
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
            // if the slot (the supposed eligible inference encoder) is not in the shards inference set
            // the verification should fail. Slots do not change with routing.
            if !shard.inference_set().contains(&signed_commit.slot()) {
                return Err(shared::error::SharedError::ValidationError("s".to_string()));
            }
            // If there exists a route, we need to verify it, otherwise skip
            if let Some(signed_route) = signed_commit.route() {
                // if the shards inference already contains the signed route this is invalid. Routing
                // may only occur to non-eligible nodes.
                if shard.inference_set().contains(&signed_route.destination())
                    // check if the route destination is the original eligible slot
                    // this redundancy is not allowed
                    || &signed_route.destination() == &signed_commit.slot()
                {
                    return Err(shared::error::SharedError::ValidationError(
                        "invalid route destination".to_string(),
                    ));
                }

                // the digest of the auth token supplied with the commit must match the digest included in the signed
                // route message. By forcing a signature of this digest, signed route messages cannot be replayed for different shards
                if signed_route.auth_token_digest() != auth_token_digest {
                    return Err(shared::error::SharedError::ValidationError("s".to_string()));
                }

                // check signature of route is by the slot
                // the original slot must sign off on the route in order to be eligible
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
            // whoever is submitting the commit whether that is the original slot or a valid route must correctly sign
            // off on the commit
            let _ = signed_commit.verify(
                Scope::ShardCommitRoute,
                self.context
                    .encoder_committee
                    .encoder(signed_commit.committer())
                    .encoder_key
                    .inner(),
            )?;
            // the metadata must be valid

            let metadata = signed_commit.commit();
            // TODO: update to actually check commit size, shape, etc according to embedding standards
            let _ = verify_metadata(None, None, None, None, None)(metadata)?;
            Ok(())
        })
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

        let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;
        let epoch = verified_commit.auth_token().epoch();
        let slot = verified_commit.slot();
        let committer = verified_commit.committer();

        // the commit is idempotently committed to the store. If there is a conflict this should
        // fail such that there is no partial signature produced that would attest to the commit
        let _ = self.store.lock_signed_commit_digest(
            epoch,
            shard_ref,
            slot,
            committer,
            verified_commit.digest(),
        )?;

        // produce the partial signature attesting to seeing the commit
        let keypair = self.encoder_keypair.inner().copy();
        let partial_sig = Signed::new(signed_commit, Scope::ShardCertificate, &keypair.private())
            .map_err(ShardError::SerializationFailure)?;

        Ok(partial_sig.serialized())
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

        let committer = self
            .context
            .encoder_committee
            .encoder(certified_commit.committer());

        let probe_metadata = committer.probe.clone();
        // perform verification on type and auth including signature checks
        let verified_certified_commit = Verified::new(
            certified_commit,
            certified_commit_bytes,
            |certified_commit| {
                // pull out the idices of the certificate
                let certifier_indices = certified_commit.indices();

                for index in &certifier_indices {
                    // for each index we need to ensure that they are members of the evaluation set
                    // evaluation sets do not change for a given shard so this works with routing
                    if !shard.evaluation_set().contains(index) {
                        return Err(shared::error::SharedError::ValidationError(
                            "index not in evaluation set".to_string(),
                        ));
                    }
                }

                // checks to ensure that the number of unique indices meets quorum and verifies the agg signature
                // using the corresponding public keys from those indices
                certified_commit
                    .verify_quorum(Scope::ShardCertificate, &self.context.encoder_committee)?;

                // It is redundant to reverify the signed commit that is certified since a quorum must be met to produce a valid certificate
                // at least a quorum number of evaluators must validate the commit, and honest assumptions are out of scope here
                Ok(())
            },
        )
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

        let _ = self
            .dispatcher
            .dispatch_certified_commit(peer, shard, probe_metadata, verified_certified_commit)
            .await?;
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
            // the voter must be a member of the evaluation set
            // evaluation sets do not change for a given shard
            if !shard.evaluation_set().contains(&votes.voter()) {
                return Err(shared::error::SharedError::ValidationError(
                    "voter is not in evaluation set".to_string(),
                ));
            }
            // verify that the reject encoder indices are valid slots
            // may want to check for uniqueness but since acceptance votes are implicit
            // and rejection votes are implicit multiple redundant votes is fine unless they are counted twice
            for index in votes.rejects() {
                if !shard.inference_set().contains(index) {
                    return Err(shared::error::SharedError::ValidationError(
                        "index not in inference set".to_string(),
                    ));
                }
            }
            // the signature of the vote message must match the voter. The inclusion of the voter in the
            // evaluation set is checked above
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
        let _ = self
            .dispatcher
            .dispatch_commit_votes(peer, verified_commit_votes)
            .await?;
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
        let slot = reveal.slot();
        let epoch = reveal.auth_token().epoch();
        let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;
        let certified_commit = self.store.get_certified_commit(epoch, shard_ref, slot)?;

        let verified_reveal = Verified::new(reveal.clone(), reveal_bytes, |reveal| {
            let encryption =
                certified_commit
                    .commit()
                    .encryption()
                    .ok_or(SharedError::ValidationError(
                        "missing encryption".to_string(),
                    ))?;

            let reveal_key_digest = Digest::new(reveal.key())?;
            if encryption.key_digest() != reveal_key_digest {
                return Err(SharedError::ValidationError(
                    "key digests do not match".to_string(),
                ));
            }
            // the reveal slot must be a member of the shard inference slot
            // in the case of routing, the original slot is still expected to handle the reveal since this allows
            // routing to take place without needing to reorganize all the communication of the shard
            if !shard.inference_set().contains(&reveal.slot()) {
                return Err(shared::error::SharedError::ValidationError(
                    "index not in inference set".to_string(),
                ));
            }
            // the reveal message must be signed by the slot
            let _ = reveal.verify(
                Scope::ShardReveal,
                self.context
                    .encoder_committee
                    .encoder(reveal.slot())
                    .encoder_key
                    .inner(),
            )?;
            Ok(())
        })
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

        let _ = self
            .dispatcher
            .dispatch_reveal(
                peer,
                shard,
                certified_commit.commit().to_owned(),
                verified_reveal,
            )
            .await?;
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

        let verified_reveal_votes = Verified::new(votes, votes_bytes, |votes| {
            // the voter must be a member of the evaluation set
            // evaluation sets do not change for a given shard
            if !shard.evaluation_set().contains(&votes.voter()) {
                return Err(shared::error::SharedError::ValidationError(
                    "voter is not in evaluation set".to_string(),
                ));
            }
            // verify that the reject encoder indices are valid slots
            // may want to check for uniqueness but since acceptance votes are implicit
            // and rejection votes are implicit multiple redundant votes is fine unless they are counted twice
            for index in votes.rejects() {
                if !shard.inference_set().contains(index) {
                    return Err(shared::error::SharedError::ValidationError(
                        "index not in inference set".to_string(),
                    ));
                }
            }
            // the signature of the vote message must match the voter. The inclusion of the voter in the
            // evaluation set is checked above
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
        let _ = self
            .dispatcher
            .dispatch_reveal_votes(peer, verified_reveal_votes)
            .await?;
        Ok(())
    }
}
