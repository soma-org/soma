use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct AdvanceEpochRequest {}

#[derive(Debug, Deserialize, Serialize)]
pub struct AdvanceEpochResponse {
    pub epoch: u64,
}
