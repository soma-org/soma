use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use parking_lot::RwLock;
use tokio_util::sync::CancellationToken;

use crate::{
    error::{ShardError, ShardResult},
    types::{context::Quorum, digest::Digest, network_committee::NetworkingIndex, shard::ShardRef},
};

pub(crate) struct Supervisor {
    // TODO: potentially switch to a B-Tree to allow for cancelling and cleaning all things that are
    // two epochs away.
    cancellation_tokens: HashMap<(ShardRef, Option<NetworkingIndex>), CancellationToken>,
    shard_size: HashMap<ShardRef, usize>,
    commit_slots: HashMap<ShardRef, HashSet<NetworkingIndex>>,
    reveal_slots: HashMap<ShardRef, HashSet<NetworkingIndex>>,
}

impl Supervisor {
    pub(crate) fn new() -> Self {
        Self {
            cancellation_tokens: HashMap::new(),
            shard_size: HashMap::new(),
            commit_slots: HashMap::new(),
            reveal_slots: HashMap::new(),
        }
    }
    pub(crate) fn create_shard(
        &mut self,
        shard_ref: ShardRef,
        shard_size: usize,
    ) -> ShardResult<CancellationToken> {
        let _ = self
            .shard_size
            .entry(shard_ref.clone())
            .or_insert(shard_size);

        Ok(self
            .cancellation_tokens
            .entry((shard_ref, None))
            .or_insert_with(CancellationToken::new)
            .clone())
    }
    pub(crate) fn shard_cancellation_token(
        &mut self,
        shard_ref: ShardRef,
        peer: Option<NetworkingIndex>,
    ) -> ShardResult<CancellationToken> {
        // Get the parent shard token first - error if it doesn't exist
        let parent_token = self
            .cancellation_tokens
            .get(&(shard_ref.clone(), None))
            .ok_or_else(|| ShardError::ShardNotFound(shard_ref.clone().to_string()))?
            .clone();

        if let Some(peer_id) = peer {
            let child_token = parent_token.child_token();

            Ok(self
                .cancellation_tokens
                .entry((shard_ref, Some(peer_id)))
                .or_insert(child_token)
                .clone())
        } else {
            Ok(parent_token)
        }
    }

    pub(crate) fn track_commit_certificate(
        &mut self,
        shard_ref: ShardRef,
        peer: NetworkingIndex,
    ) -> ShardResult<()> {
        let size = self
            .shard_size
            .get(&shard_ref)
            .ok_or_else(|| ShardError::ShardNotFound(shard_ref.clone().to_string()))?;

        let commit_slots = self.commit_slots.entry(shard_ref).or_default();

        commit_slots.insert(peer);

        if commit_slots.len() >= *size {
            // trigger
        }

        Ok(())
    }

    pub(crate) fn track_reveal_certificate(
        &mut self,
        shard_ref: ShardRef,
        peer: NetworkingIndex,
    ) -> ShardResult<()> {
        let size = self
            .shard_size
            .get(&shard_ref)
            .ok_or_else(|| ShardError::ShardNotFound(shard_ref.clone().to_string()))?;

        let reveal_slots = self.reveal_slots.entry(shard_ref).or_default();

        reveal_slots.insert(peer);

        if reveal_slots.len() >= *size {
            // trigger
        }

        Ok(())
    }
}
