use crate::{
    block::{BlockRef, VerifiedBlock},
    commit::{CommitRange, TrustedCommit},
    context::Context,
    error::ConsensusResult,
    Round,
};
use bytes::Bytes;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use types::committee::AuthorityIndex;
use types::crypto::NetworkKeyPair;

// Tonic generated RPC stubs.
mod tonic_gen {
    include!("proto/consensus.ConsensusService.rs");
}

#[cfg(all(test))]
mod network_tests;
#[cfg(test)]
pub(crate) mod test_network;

pub(crate) mod tonic_network;

#[async_trait]
pub(crate) trait NetworkClient: Send + Sync + Sized + 'static {
    /// Sends a serialized SignedBlock to a peer.
    async fn send_block(
        &self,
        peer: AuthorityIndex,
        block: &VerifiedBlock,
        timeout: Duration,
    ) -> ConsensusResult<()>;

    /// Fetches serialized `SignedBlock`s from a peer. It also might return additional ancestor blocks
    /// of the requested blocks according to the provided `highest_accepted_rounds`. The `highest_accepted_rounds`
    /// length should be equal to the committee size. If `highest_accepted_rounds` is empty then it will
    /// be simply ignored.
    async fn fetch_blocks(
        &self,
        peer: AuthorityIndex,
        block_refs: Vec<BlockRef>,
        highest_accepted_rounds: Vec<Round>,
        timeout: Duration,
    ) -> ConsensusResult<Vec<Bytes>>;

    /// Fetches serialized commits in the commit range from a peer.
    /// Returns a tuple of both the serialized commits, and serialized blocks that contain
    /// votes certifying the last commit.
    async fn fetch_commits(
        &self,
        peer: AuthorityIndex,
        commit_range: CommitRange,
        timeout: Duration,
    ) -> ConsensusResult<(Vec<Bytes>, Vec<Bytes>)>;

    /// Fetches the latest block from `peer` for the requested `authorities`. The latest blocks
    /// are returned in the serialised format of `SignedBlocks`. The method can return multiple
    /// blocks per peer as its possible to have equivocations.
    async fn fetch_latest_blocks(
        &self,
        peer: AuthorityIndex,
        authorities: Vec<AuthorityIndex>,
        timeout: Duration,
    ) -> ConsensusResult<Vec<Bytes>>;
}

/// Network service for handling requests from peers.
#[async_trait]
pub(crate) trait NetworkService: Send + Sync + 'static {
    /// Handles the block sent from the peer via either unicast RPC or subscription stream.
    /// Peer value can be trusted to be a valid authority index.
    /// But serialized_block must be verified before its contents are trusted.
    async fn handle_send_block(&self, peer: AuthorityIndex, block: Bytes) -> ConsensusResult<()>;

    /// Handles the request to fetch blocks by references from the peer.
    async fn handle_fetch_blocks(
        &self,
        peer: AuthorityIndex,
        block_refs: Vec<BlockRef>,
        highest_accepted_rounds: Vec<Round>,
    ) -> ConsensusResult<Vec<Bytes>>;

    /// Handles the request to fetch commits by index range from the peer.
    async fn handle_fetch_commits(
        &self,
        peer: AuthorityIndex,
        commit_range: CommitRange,
    ) -> ConsensusResult<(Vec<TrustedCommit>, Vec<VerifiedBlock>)>;

    /// Handles the request to fetch the latest block for the provided `authorities`.
    async fn handle_fetch_latest_blocks(
        &self,
        peer: AuthorityIndex,
        authorities: Vec<AuthorityIndex>,
    ) -> ConsensusResult<Vec<Bytes>>;
}

pub(crate) trait NetworkManager<S>: Send + Sync
where
    S: NetworkService,
{
    type Client: NetworkClient;

    /// Creates a new network manager.
    fn new(context: Arc<Context>, network_keypair: NetworkKeyPair) -> Self;

    /// Returns the network client.
    fn client(&self) -> Arc<Self::Client>;

    /// Installs network service.
    async fn install_service(&mut self, service: Arc<S>);

    /// Stops the network service.
    async fn stop(&mut self);
}
