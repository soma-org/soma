pub mod modules;
pub mod probe;

const V1_EMBEDDING_DIM: usize = 4_096;
const V1_VOCAB_SIZE: usize = 256;
const V1_NUM_HEADS: usize = 32;
const V1_NUM_LAYERS: usize = 6;
const V1_MAX_SEQ_LEN: usize = 512;
const V1_PWFF_HIDDEN_DIM: usize = 16_384;
const V1_MAX_WAVELENGTH: usize = 10_000;
