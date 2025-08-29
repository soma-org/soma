use std::{sync::Arc, time::Duration};

use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::Store,
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        commit::{Commit, CommitV1},
        input::{Input, InputAPI},
        reveal::{Reveal, RevealV1},
    },
};
use async_trait::async_trait;
use evaluation::{
    messaging::EvaluationClient, EvaluationInput, EvaluationInputV1, EvaluationOutputAPI,
};
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use inference::{
    client::InferenceClient, InferenceInput, InferenceInputV1, InferenceOutput, InferenceOutputAPI,
};
use objects::{
    networking::{downloader::Downloader, ObjectNetworkClient},
    storage::{ObjectPath, ObjectStorage},
};
use shared::{
    actors::{ActorHandle, ActorMessage, Processor},
    crypto::keys::{EncoderKeyPair, PeerPublicKey},
    digest::Digest,
    error::{ShardError, ShardResult},
    metadata::{
        DownloadableMetadata, DownloadableMetadataAPI, DownloadableMetadataV1, Metadata,
        MetadataAPI,
    },
    scope::Scope,
    shard::Shard,
    signed::Signed,
    verified::Verified,
};
use soma_network::multiaddr::Multiaddr;
use tracing::info;

use super::commit::CommitProcessor;

pub(crate) struct InputProcessor<
    C: EncoderInternalNetworkClient,
    O: ObjectNetworkClient,
    M: InferenceClient,
    S: ObjectStorage,
    P: EvaluationClient,
> {
    store: Arc<dyn Store>,
    downloader: ActorHandle<Downloader<O, S>>,
    broadcaster: Arc<Broadcaster<C>>,
    inference_client: Arc<M>,
    evaluation_client: Arc<P>,
    encoder_keypair: Arc<EncoderKeyPair>,
    storage: Arc<S>,
    commit_pipeline: ActorHandle<CommitProcessor<O, C, S, P>>,
}

impl<
        C: EncoderInternalNetworkClient,
        O: ObjectNetworkClient,
        M: InferenceClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > InputProcessor<C, O, M, S, P>
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        store: Arc<dyn Store>,
        downloader: ActorHandle<Downloader<O, S>>,
        broadcaster: Arc<Broadcaster<C>>,
        inference_client: Arc<M>,
        evaluation_client: Arc<P>,
        encoder_keypair: Arc<EncoderKeyPair>,
        storage: Arc<S>,
        commit_pipeline: ActorHandle<CommitProcessor<O, C, S, P>>,
    ) -> Self {
        Self {
            store,
            downloader,
            broadcaster,
            inference_client,
            evaluation_client,
            encoder_keypair,
            storage,
            commit_pipeline,
        }
    }
}

#[async_trait]
impl<
        C: EncoderInternalNetworkClient,
        O: ObjectNetworkClient,
        M: InferenceClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > Processor for InputProcessor<C, O, M, S, P>
{
    type Input = (
        Shard,
        Verified<Signed<Input, min_sig::BLS12381Signature>>,
        PeerPublicKey,
        Multiaddr,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let keypair = self.encoder_keypair.inner().copy();
        let result: ShardResult<()> = async {
            let (shard, verified_signed_input, peer, address) = msg.input;
            let epoch = shard.epoch();
            let downloadable_metadata = verified_signed_input
                .auth_token()
                .metadata_commitment()
                .downloadable_metadata();

            let metadata = downloadable_metadata.metadata();

            self.downloader
                .process(downloadable_metadata, msg.cancellation.clone())
                .await?;

            let inference_input = InferenceInput::V1(InferenceInputV1::new(metadata.clone()));

            let inference_timeout = Duration::from_secs(1);
            let inference_output = self
                .inference_client
                .call(inference_input, inference_timeout)
                .await
                .map_err(ShardError::InferenceError)?;

            let evaluation_input = EvaluationInput::V1(EvaluationInputV1::new(
                metadata,
                inference_output.embeddings(),
                inference_output.probe_set(),
            ));
            let evaluation_timeout = Duration::from_secs(1);

            let evaluation_output = self
                .evaluation_client
                .evaluation(evaluation_input, evaluation_timeout)
                .await
                .map_err(ShardError::EvaluationError)?;

            // send input data object path to inference
            // inference returns: probe_set, object path to representations/byte-ranges
            // send input data, probe set, and representations to evaluation
            // evaluation returns: score and summary embedding bytes
            // create and sign a reveal message
            // store in datastore
            // use the digest of the reveal to create and sign a commit message

            let embedding_metadata = match inference_output.embeddings() {
                Metadata::V1(m) => m,
            };

            let tensors = DownloadableMetadata::V1(DownloadableMetadataV1::new(
                peer,
                address,
                embedding_metadata, // use the underlying MetadataV1 rather than wrapping in the Metadata enum
            ));

            let reveal = Reveal::V1(RevealV1::new(
                verified_signed_input.auth_token().clone(),
                self.encoder_keypair.public(),
                evaluation_output.score(),
                inference_output.probe_set(),
                tensors,
                evaluation_output.summary_embedding(),
            ));

            let inner_keypair = self.encoder_keypair.inner().copy();
            let signed_reveal =
                Signed::new(reveal, Scope::Reveal, &inner_keypair.private()).unwrap();

            let signed_reveal_digest =
                Digest::new(&signed_reveal).map_err(ShardError::DigestFailure)?;

            let verified_reveal = Verified::from_trusted(signed_reveal).unwrap();
            self.store.add_signed_reveal(&shard, &verified_reveal)?;

            let commit = Commit::V1(CommitV1::new(
                verified_signed_input.auth_token().clone(),
                self.encoder_keypair.public(),
                signed_reveal_digest,
            ));

            let inner_keypair = self.encoder_keypair.inner().copy();
            // Sign the commit
            let signed_commit =
                Signed::new(commit, Scope::Commit, &inner_keypair.private()).unwrap();
            let verified = Verified::from_trusted(signed_commit).unwrap();

            self.commit_pipeline
                .process((shard.clone(), verified.clone()), msg.cancellation.clone())
                .await?;
            info!("Broadcasting to other nodes");
            // Broadcast to other encoders
            self.broadcaster
                .broadcast(
                    verified.clone(),
                    shard.encoders(),
                    |client, peer, verified_type| async move {
                        client
                            .send_commit(&peer, &verified_type, MESSAGE_TIMEOUT)
                            .await?;
                        Ok(())
                    },
                )
                .await?;

            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
