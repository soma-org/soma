use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shard {
    pub quorum_threshold: u32,
    pub encoders: Vec<Vec<u8>>, // Raw bytes for encoder public keys
    pub seed: Vec<u8>,          // Raw bytes for the seed digest
    pub epoch: u64,
}

impl Shard {
    pub fn new(quorum_threshold: u32, encoders: Vec<Vec<u8>>, seed: Vec<u8>, epoch: u64) -> Self {
        Self {
            quorum_threshold,
            encoders,
            seed,
            epoch,
        }
    }
}
