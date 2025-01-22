use serde::{Deserialize, Serialize};
use shared::crypto::keys::ProtocolPublicKey;


/// Score is an encoder score for a given shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Score {
    /// the identity of the encoder
    identity: ProtocolPublicKey,
    /// the score
    score: f64,
}

impl Score {
    /// creates a new score given a public key and score
    pub const fn new(identity: ProtocolPublicKey, score: f64) -> Self {
        Self { identity, score }
    }
}
