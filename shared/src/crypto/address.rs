use serde::{Deserialize, Serialize};

const ADDRESS_LENGTH: usize = 32;

#[derive(Debug, Default, PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize)]
pub struct Address([u8; ADDRESS_LENGTH]);
