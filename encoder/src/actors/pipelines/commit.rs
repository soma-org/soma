use crate::{
    actors::{
        workers::downloader::{Downloader, DownloaderInput},
        ActorHandle, ActorMessage, Processor,
    },
    core::shard_tracker::ShardTracker,
    datastore::Store,
    error::ShardResult,
    messaging::EncoderInternalNetworkClient,
    types::{
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
    },
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use objects::{networking::ObjectNetworkClient, storage::ObjectStorage};
use shared::{
    crypto::keys::PeerPublicKey, probe::ProbeMetadata, signed::Signed, verified::Verified,
};
use soma_network::multiaddr::Multiaddr;
use std::sync::Arc;
use tracing::info;

pub(crate) struct CommitProcessor<
    E: EncoderInternalNetworkClient,
    C: ObjectNetworkClient,
    S: ObjectStorage,
> {
    store: Arc<dyn Store>,
    shard_tracker: Arc<ShardTracker<E, S>>,
    downloader: ActorHandle<Downloader<C, S>>,
}

impl<E: EncoderInternalNetworkClient, C: ObjectNetworkClient, S: ObjectStorage>
    CommitProcessor<E, C, S>
{
    pub(crate) fn new(
        store: Arc<dyn Store>,
        shard_tracker: Arc<ShardTracker<E, S>>,
        downloader: ActorHandle<Downloader<C, S>>,
    ) -> Self {
        Self {
            store,
            shard_tracker,
            downloader,
        }
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient, O: ObjectNetworkClient, S: ObjectStorage> Processor
    for CommitProcessor<E, O, S>
{
    type Input = (
        Shard,
        Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        ProbeMetadata,
        PeerPublicKey,
        Multiaddr,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, verified_signed_commit, probe_metadata, peer, address) = msg.input;

            let commit_input = DownloaderInput::new(
                peer.clone(),
                address.clone(),
                verified_signed_commit.commit_metadata().clone(),
            );

            self.downloader
                .process(commit_input, msg.cancellation.clone())
                .await?;

            // let probe_input = DownloaderInput::new(peer, address, probe_metadata.metadata());

            // self.downloader
            //     .process(probe_input, msg.cancellation.clone())
            //     .await?;

            let _ = self
                .store
                .add_signed_commit(&shard, &verified_signed_commit)?;

            self.shard_tracker
                .track_valid_commit(shard, verified_signed_commit)
                .await?;

            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
