use async_trait::async_trait;
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use probe::messaging::ProbeClient;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::{
    actors::{ActorMessage, Processor},
    core::{internal_broadcaster::Broadcaster, shard_tracker::ShardTracker},
    datastore::Store,
    error::{ShardError, ShardResult},
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        shard::Shard,
        shard_commit::{Route, ShardCommit, ShardCommitAPI},
        shard_commit_votes::{ShardCommitVotes, ShardCommitVotesV1},
        shard_reveal::{ShardReveal, ShardRevealAPI, ShardRevealV1},
        shard_reveal_votes::{ShardRevealVotes, ShardRevealVotesV1},
        shard_scores::{ScoreSet, ShardScores, ShardScoresV1},
        shard_verifier::ShardAuthToken,
    },
};
use objects::storage::ObjectStorage;
use shared::{
    crypto::{
        keys::{EncoderKeyPair, EncoderPublicKey},
        Aes256IV, Aes256Key, EncryptionKey,
    },
    digest::Digest,
    metadata::Metadata,
    scope::Scope,
    signed::Signed,
    verified::Verified,
};

/// Action types that the ShardBroadcastProcessor can handle
#[derive(Debug)]
pub(crate) enum BroadcastAction {
    Commit(Shard, ShardAuthToken, Option<Route>, Metadata),
    CommitVote(ShardAuthToken, Shard),
    Reveal(ShardAuthToken, Shard),
    RevealVote(ShardAuthToken, Shard),
    Scores(Shard, ShardAuthToken, ScoreSet),
}

/// Processor for handling all broadcast operations
pub(crate) struct BroadcastProcessor<E: EncoderInternalNetworkClient> {
    broadcaster: Arc<Broadcaster<E>>,
    encoder_keypair: Arc<EncoderKeyPair>,
}

impl<E: EncoderInternalNetworkClient> BroadcastProcessor<E> {
    pub(crate) fn new(
        broadcaster: Arc<Broadcaster<E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
    ) -> Self {
        Self {
            broadcaster,
            encoder_keypair,
        }
    }

    /// Implementation of commit broadcast
    async fn handle_commit(
        &self,
        shard: Shard,
        auth_token: ShardAuthToken,
        route: Option<Route>,
        metadata: Metadata,
    ) -> ShardResult<()> {
        info!("Handling commit in BroadcastProcessor");
        let inner_keypair = self.encoder_keypair.inner().copy();

        // Create signed route if provided
        let signed_route = route.map(|r| {
            Signed::new(r, Scope::ShardCommitRoute, &inner_keypair.copy().private()).unwrap()
        });

        // Create commit
        let commit = ShardCommit::new_v1(
            auth_token,
            self.encoder_keypair.public(),
            signed_route,
            metadata,
        );

        // Sign the commit
        let signed_commit =
            Signed::new(commit, Scope::ShardCommit, &inner_keypair.private()).unwrap();
        let verified = Verified::from_trusted(signed_commit).unwrap();

        info!("Broadcasting to other nodes");
        // Broadcast to other encoders
        let res = self
            .broadcaster
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
            .await;

        if let Err(e) = res {
            tracing::error!("Error broadcasting commits: {:?}", e);
        }

        let store = self.store.clone();
        let shard_tracker = self.shard_tracker.clone();

        tokio::spawn(async move {
            // Process locally first
            info!("Adding to store");
            if let Err(e) = store.add_signed_commit(&shard, &verified) {
                tracing::error!("Error adding signed commit to store: {:?}", e);
            }

            info!("Tracking valid commit in ShardTracker");
            if let Err(e) = shard_tracker
                .track_valid_commit(shard.clone(), verified.clone())
                .await
            {
                tracing::error!("Error tracking valid commit: {:?}", e);
            }
        });

        Ok(())
    }

    /// Implementation of commit vote broadcast
    async fn handle_commit_vote(
        &self,
        auth_token: ShardAuthToken,
        shard: Shard,
    ) -> ShardResult<()> {
        // Get commits from store
        let commits = match self.store.get_signed_commits(&shard) {
            Ok(commits) => commits,
            Err(e) => {
                tracing::error!("Error getting signed commits: {:?}", e);
                return Err(ShardError::MissingData);
            }
        };

        // Format commits for votes
        let commits = commits
            .iter()
            .map(|commit| (commit.encoder().clone(), Digest::new(commit).unwrap()))
            .collect();

        // Create votes
        let votes = ShardCommitVotes::V1(ShardCommitVotesV1::new(
            auth_token,
            self.encoder_keypair.public(),
            commits,
        ));

        // Sign votes
        let inner_keypair = self.encoder_keypair.inner().copy();
        let signed_votes =
            Signed::new(votes, Scope::ShardCommitVotes, &inner_keypair.private()).unwrap();
        let verified = Verified::from_trusted(signed_votes).unwrap();

        info!(
            "Broadcasting commit vote to other encoders: {:?}",
            shard.encoders()
        );

        // Broadcast to other encoders
        let res = self
            .broadcaster
            .broadcast(
                verified.clone(),
                shard.encoders(),
                |client, peer, verified_type| async move {
                    client
                        .send_commit_votes(&peer, &verified_type, MESSAGE_TIMEOUT)
                        .await?;
                    Ok(())
                },
            )
            .await;

        if let Err(e) = res {
            tracing::error!("Error broadcasting commit votes: {:?}", e);
        }

        let store = self.store.clone();
        let shard_tracker = self.shard_tracker.clone();

        tokio::spawn(async move {
            // Process locally
            if let Err(e) = store.add_commit_votes(&shard, &verified) {
                tracing::error!("Error adding signed commit votes to store: {:?}", e);
            }

            if let Err(e) = shard_tracker
                .track_valid_commit_votes(shard.clone(), verified.clone())
                .await
            {
                tracing::error!("Error tracking valid commit votes: {:?}", e);
            }
        });

        Ok(())
    }

    /// Implementation of reveal broadcast
    async fn handle_reveal(&self, auth_token: ShardAuthToken, shard: Shard) -> ShardResult<()> {
        info!("Beginning to broadcast reveal");
        // Generate key from signature over shard
        let inner_keypair = self.encoder_keypair.inner().copy();
        let signed_shard = Signed::new(
            shard.clone(),
            Scope::EncryptionKey,
            &inner_keypair.copy().private(),
        )
        .unwrap();

        let signature_bytes = signed_shard.raw_signature();

        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&signature_bytes[..32]); // Use only the first 32 bytes

        let mut iv_bytes = [0u8; 16];
        iv_bytes.copy_from_slice(&signature_bytes[..16]); // Use only the first 16 bytes

        let key = EncryptionKey::Aes256(Aes256IV {
            iv: iv_bytes,
            key: Aes256Key::from(key_bytes),
        });

        // Create reveal
        let reveal = ShardReveal::V1(ShardRevealV1::new(
            auth_token,
            self.encoder_keypair.public(),
            key,
        ));

        // Sign reveal
        let signed_reveal =
            Signed::new(reveal, Scope::ShardReveal, &inner_keypair.private()).unwrap();
        let verified = Verified::from_trusted(signed_reveal).unwrap();

        info!("Broadcasting reveal to other nodes");

        // Broadcast to other encoders
        let res = self
            .broadcaster
            .broadcast(
                verified.clone(),
                shard.encoders(),
                |client, peer, verified_type| async move {
                    client
                        .send_reveal(&peer, &verified_type, MESSAGE_TIMEOUT)
                        .await?;
                    Ok(())
                },
            )
            .await;

        if let Err(e) = res {
            tracing::error!("Error broadcasting reveal: {:?}", e);
        }

        let store = self.store.clone();
        let shard_tracker = self.shard_tracker.clone();

        tokio::spawn(async move {
            // Process locally first
            if let Err(e) = store.add_signed_reveal(&shard, &verified) {
                tracing::error!("Error adding signed reveal to store: {:?}", e);
            }

            if let Err(e) = shard_tracker
                .track_valid_reveal(shard.clone(), verified.clone())
                .await
            {
                tracing::error!("Error tracking valid reveal: {:?}", e);
            }
        });

        Ok(())
    }

    /// Implementation of reveal vote broadcast
    async fn handle_reveal_vote(
        &self,
        auth_token: ShardAuthToken,
        shard: Shard,
    ) -> ShardResult<()> {
        // Get reveals from store
        let reveals = match self.store.get_signed_reveals(&shard) {
            Ok(reveals) => reveals,
            Err(e) => {
                tracing::error!("Error getting signed reveals: {:?}", e);
                return Err(ShardError::MissingData);
            }
        };

        // Format reveals for votes
        let reveals = reveals
            .iter()
            .map(|reveal| reveal.encoder().clone())
            .collect();

        // Create votes
        let votes = ShardRevealVotes::V1(ShardRevealVotesV1::new(
            auth_token,
            self.encoder_keypair.public(),
            reveals,
        ));

        // Sign votes
        let inner_keypair = self.encoder_keypair.inner().copy();
        let signed_votes =
            Signed::new(votes, Scope::ShardRevealVotes, &inner_keypair.private()).unwrap();
        let verified = Verified::from_trusted(signed_votes).unwrap();

        // Broadcast to other encoders
        let res = self
            .broadcaster
            .broadcast(
                verified.clone(),
                shard.encoders(),
                |client, peer, verified_type| async move {
                    client
                        .send_reveal_votes(&peer, &verified_type, MESSAGE_TIMEOUT)
                        .await?;
                    Ok(())
                },
            )
            .await;

        if let Err(e) = res {
            tracing::error!("Error broadcasting reveal votes: {:?}", e);
        }

        let store = self.store.clone();
        let shard_tracker = self.shard_tracker.clone();

        tokio::spawn(async move {
            // Process locally first
            if let Err(e) = store.add_reveal_votes(&shard, &verified) {
                tracing::error!("Error adding signed reveal votes to store: {:?}", e);
            }

            if let Err(e) = shard_tracker
                .track_valid_reveal_votes(shard.clone(), verified.clone())
                .await
            {
                tracing::error!("Error tracking valid reveal votes: {:?}", e);
            }
        });

        Ok(())
    }

    /// Implementation of scores broadcast
    async fn handle_scores(
        &self,
        shard: Shard,
        auth_token: ShardAuthToken,
        score_set: ScoreSet,
    ) -> ShardResult<()> {
        let inner_keypair = self.encoder_keypair.inner().copy();

        // Sign score set
        let signed_score_set = Signed::new(
            score_set,
            Scope::ShardScores,
            &inner_keypair.copy().private(),
        )
        .unwrap();

        // Create scores
        let scores = ShardScores::V1(ShardScoresV1::new(
            auth_token,
            self.encoder_keypair.public(),
            signed_score_set,
        ));

        // Sign scores
        let signed_scores =
            Signed::new(scores, Scope::ShardScores, &inner_keypair.private()).unwrap();
        let verified = Verified::from_trusted(signed_scores).unwrap();

        // Broadcast to other encoders
        let res = self
            .broadcaster
            .broadcast(
                verified.clone(),
                shard.encoders(),
                |client, peer, verified_type| async move {
                    client
                        .send_scores(&peer, &verified_type, MESSAGE_TIMEOUT)
                        .await?;
                    Ok(())
                },
            )
            .await;

        if let Err(e) = res {
            tracing::error!("Error broadcasting scores: {:?}", e);
        }

        let store = self.store.clone();
        let shard_tracker = self.shard_tracker.clone();

        tokio::spawn(async move {
            // Process locally first
            if let Err(e) = store.add_signed_scores(&(shard.clone()), &(verified.clone())) {
                tracing::error!("Error adding signed scores to store: {:?}", e);
            }

            if let Err(e) = shard_tracker
                .track_valid_scores(shard.clone(), verified.clone())
                .await
            {
                tracing::error!("Error tracking valid scores: {:?}", e);
            }
        });

        Ok(())
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient> Processor for BroadcastProcessor<E> {
    type Input = BroadcastAction;
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result = match msg.input {
            BroadcastAction::Commit(shard, auth_token, route, metadata) => {
                self.handle_commit(shard, auth_token, route, metadata).await
            }
            BroadcastAction::CommitVote(auth_token, shard) => {
                self.handle_commit_vote(auth_token, shard).await
            }
            BroadcastAction::Reveal(auth_token, shard) => {
                info!("Time to reveal!");
                self.handle_reveal(auth_token, shard).await
            }
            BroadcastAction::RevealVote(auth_token, shard) => {
                self.handle_reveal_vote(auth_token, shard).await
            }
            BroadcastAction::Scores(shard, auth_token, score_set) => {
                self.handle_scores(shard, auth_token, score_set).await
            }
        };

        if let Err(e) = &result {
            tracing::error!("Error in broadcast processor: {:?}", e);
        }

        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {
        // No special cleanup needed
    }
}
