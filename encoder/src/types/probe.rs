use bytes::Bytes;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Probe {
    weights: Bytes,
}
