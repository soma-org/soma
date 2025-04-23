use std::future::Future;
use std::sync::Arc;

use fastcrypto::traits::KeyPair;
use shared::{
    crypto::{
        keys::{EncoderKeyPair, EncoderPublicKey},
        Aes256IV, Aes256Key, EncryptionKey,
    },
    digest::Digest,
    scope::Scope,
    signed::Signed,
    verified::Verified,
};

use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::Store,
    types::{
        shard::Shard,
        shard_commit::ShardCommitAPI,
        shard_commit_votes::{ShardCommitVotes, ShardCommitVotesV1},
        shard_reveal::{ShardReveal, ShardRevealAPI, ShardRevealV1},
        shard_reveal_votes::{ShardRevealVotes, ShardRevealVotesV1},
        shard_scores::{ScoreSet, ShardScores, ShardScoresV1},
        shard_verifier::ShardAuthToken,
    },
};

use super::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT};

pub(crate) fn broadcast_commit_vote<C: EncoderInternalNetworkClient + 'static>(
    peers: Vec<EncoderPublicKey>,
    broadcaster: Arc<Broadcaster<C>>,
    store: Arc<dyn Store>,
    auth_token: ShardAuthToken,
    shard: Shard,
    keypair: Arc<EncoderKeyPair>,
) -> impl FnOnce() + Send + 'static {
    move || {
        tokio::spawn(async move {
            let commits = match store.get_signed_commits(&shard) {
                Ok(commits) => commits,
                Err(_) => return, // Early return on error
            };

            let commits = commits
                .iter()
                .map(|commit| (commit.encoder().clone(), Digest::new(commit).unwrap()))
                .collect();

            let votes: ShardCommitVotes = ShardCommitVotes::V1(ShardCommitVotesV1::new(
                auth_token,
                keypair.public(),
                commits,
            ));

            let inner_keypair = keypair.inner().copy();
            let signed_votes =
                Signed::new(votes, Scope::ShardCommitVotes, &inner_keypair.private()).unwrap();
            let verified = Verified::from_trusted(signed_votes).unwrap();

            let _ = broadcaster
                .broadcast(verified, peers, |client, peer, verified_type| async move {
                    client
                        .send_commit_votes(&peer, &verified_type, MESSAGE_TIMEOUT)
                        .await;
                    Ok(())
                })
                .await;
        });
    }
}

pub(crate) fn broadcast_reveal_vote<C: EncoderInternalNetworkClient + 'static>(
    peers: Vec<EncoderPublicKey>,
    broadcaster: Arc<Broadcaster<C>>,
    store: Arc<dyn Store>,
    auth_token: ShardAuthToken,
    shard: Shard,
    keypair: Arc<EncoderKeyPair>,
) -> impl FnOnce() + Send + 'static {
    move || {
        tokio::spawn(async move {
            let reveals = match store.get_signed_reveals(&shard) {
                Ok(reveals) => reveals,
                Err(_) => return, // Early return on error
            };

            let reveals = reveals
                .iter()
                .map(|reveal| reveal.encoder().clone())
                .collect();

            let votes: ShardRevealVotes = ShardRevealVotes::V1(ShardRevealVotesV1::new(
                auth_token,
                keypair.public(),
                reveals,
            ));

            let inner_keypair = keypair.inner().copy();
            let signed_votes =
                Signed::new(votes, Scope::ShardRevealVotes, &inner_keypair.private()).unwrap();
            let verified = Verified::from_trusted(signed_votes).unwrap();

            let _ = broadcaster
                .broadcast(verified, peers, |client, peer, verified_type| async move {
                    client
                        .send_reveal_votes(&peer, &verified_type, MESSAGE_TIMEOUT)
                        .await;
                    Ok(())
                })
                .await;
        });
    }
}

pub(crate) fn broadcast_reveal<C: EncoderInternalNetworkClient + 'static>(
    peers: Vec<EncoderPublicKey>,
    broadcaster: Arc<Broadcaster<C>>,
    auth_token: ShardAuthToken,
    shard: Shard,
    keypair: Arc<EncoderKeyPair>,
) -> impl FnOnce() + Send + 'static {
    move || {
        tokio::spawn(async move {
            let inner_keypair = keypair.inner().copy();
            let signed_shard = Signed::new(
                shard.clone(),
                Scope::EncryptionKey,
                &inner_keypair.copy().private(),
            )
            .unwrap();
            let signature_bytes = signed_shard.raw_signature();

            let mut key_bytes = [0u8; 32];
            key_bytes.copy_from_slice(&signature_bytes);

            let mut iv_bytes = [0u8; 16];
            iv_bytes.copy_from_slice(&signature_bytes);

            let key = EncryptionKey::Aes256(Aes256IV {
                iv: iv_bytes,
                key: Aes256Key::from(key_bytes),
            });

            let reveal: ShardReveal =
                ShardReveal::V1(ShardRevealV1::new(auth_token, keypair.public(), key));

            let signed_reveal =
                Signed::new(reveal, Scope::ShardRevealVotes, &inner_keypair.private()).unwrap();
            let verified = Verified::from_trusted(signed_reveal).unwrap();

            let _ = broadcaster
                .broadcast(verified, peers, |client, peer, verified_type| async move {
                    client
                        .send_reveal(&peer, &verified_type, MESSAGE_TIMEOUT)
                        .await;
                    Ok(())
                })
                .await;
        });
    }
}
pub(crate) fn broadcast_scores<C: EncoderInternalNetworkClient + 'static>(
    shard: Shard,
    broadcaster: Arc<Broadcaster<C>>,
    auth_token: ShardAuthToken,
    score_set: ScoreSet,
    keypair: Arc<EncoderKeyPair>,
) -> impl FnOnce() + Send + 'static {
    move || {
        tokio::spawn(async move {
            let inner_keypair = keypair.inner().copy();

            let signed_score_set = Signed::new(
                score_set,
                Scope::ShardScores,
                &inner_keypair.copy().private(),
            )
            .unwrap();
            let scores = ShardScores::V1(ShardScoresV1::new(
                auth_token,
                keypair.public(),
                signed_score_set,
            ));

            let signed_scores =
                Signed::new(scores, Scope::ShardScores, &inner_keypair.private()).unwrap();

            let verified = Verified::from_trusted(signed_scores).unwrap();

            let _ = broadcaster
                .broadcast(
                    verified,
                    shard.encoders(),
                    |client, peer, verified_type| async move {
                        client
                            .send_scores(&peer, &verified_type, MESSAGE_TIMEOUT)
                            .await;
                        Ok(())
                    },
                )
                .await;
        });
    }
}
