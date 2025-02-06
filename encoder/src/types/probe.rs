use bytes::Bytes;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Probe {
    weights: Bytes,
}

impl Probe {
    pub(crate) fn new(weights: Bytes) -> Self {
        Self { weights }
    }
}
