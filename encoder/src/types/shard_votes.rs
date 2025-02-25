use std::marker::PhantomData;

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use super::{encoder_committee::EncoderIndex, shard_verifier::ShardAuthToken};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct CommitRound;
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct RevealRound;

/// Reject votes are explicit. The rest of encoders in a shard receive implicit accept votes.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ShardVotesAPI<T>)]
pub enum ShardVotes<T> {
    V1(ShardVotesV1<T>),
}

#[enum_dispatch]
pub trait ShardVotesAPI<T> {
    fn voter(&self) -> EncoderIndex;
    fn auth_token(&self) -> &ShardAuthToken;
    fn rejects(&self) -> &[EncoderIndex];
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ShardVotesV1<T> {
    /// stateless auth + stops replay attacks
    auth_token: ShardAuthToken,
    voter: EncoderIndex,
    /// Reject votes are explicit. The rest of encoders in a shard receive implicit accept votes.
    rejects: Vec<EncoderIndex>,
    // type marker see `CommitRound` and `RevealRound`
    marker: PhantomData<T>,
}

impl<T> ShardVotesV1<T> {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        voter: EncoderIndex,
        rejects: Vec<EncoderIndex>,
    ) -> Self {
        Self {
            auth_token,
            voter,
            rejects,
            marker: PhantomData,
        }
    }
}

impl<T> ShardVotesAPI<T> for ShardVotesV1<T> {
    fn voter(&self) -> EncoderIndex {
        self.voter
    }
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn rejects(&self) -> &[EncoderIndex] {
        &self.rejects
    }
}
