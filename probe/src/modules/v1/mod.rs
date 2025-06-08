pub(crate) mod decoder;
pub(crate) mod predictor;

const PATCH_EMBEDDING_DIM: usize = 2048; // big impact
const BYTE_EMBEDDING_DIM: usize = 512; // big impact
const VOCAB_SIZE: usize = 259; // minimal impact, scales linearly (256 + bos + eos + pad)
const NUM_HEADS: usize = 32; // minimal impact (32 matches eva byte)
const NUM_LAYERS: usize = 6; // big impact (vit B is 12)
const MAX_SEQ_LEN: usize = 256; // minimal impact, scales linearly
const HIDDEN_MULTIPLIER: usize = 4; // big impact
