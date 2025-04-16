use crate::{
    actors::{
        workers::downloader::{Downloader, DownloaderInput},
        ActorHandle, ActorMessage, Processor,
    },
    core::shard_tracker::ShardTracker,
    datastore::Store,
    error::ShardResult,
    types::{
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
    },
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use objects::{networking::ObjectNetworkClient, storage::ObjectStorage};
use shared::{crypto::keys::PeerPublicKey, signed::Signed, verified::Verified};
use soma_network::multiaddr::Multiaddr;
use std::sync::Arc;

pub(crate) struct CommitProcessor<C: ObjectNetworkClient, S: ObjectStorage> {
    store: Arc<dyn Store>,
    shard_tracker: ShardTracker,
    downloader: ActorHandle<Downloader<C, S>>,
}

impl<C: ObjectNetworkClient, S: ObjectStorage> CommitProcessor<C, S> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        shard_tracker: ShardTracker,
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
impl<O: ObjectNetworkClient, S: ObjectStorage> Processor for CommitProcessor<O, S> {
    type Input = (
        Shard,
        Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        PeerPublicKey,
        Multiaddr,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, verified_signed_commit, peer, address) = msg.input;

            let input = DownloaderInput::new(
                peer,
                address,
                verified_signed_commit.commit_metadata().clone(),
            );

            self.downloader
                .process(input, msg.cancellation.clone())
                .await?;

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
