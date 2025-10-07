pub mod modules;
pub mod probe;

const V1_EMBEDDING_DIM: usize = 1024;
const V1_VOCAB_SIZE: usize = 256;
const V1_NUM_HEADS: usize = 8;
const V1_NUM_LAYERS: usize = 4;
const V1_MAX_SEQ_LEN: usize = 256;
const V1_PWFF_HIDDEN_DIM: usize = V1_EMBEDDING_DIM * 4;
const V1_MAX_WAVELENGTH: f32 = 100_000.0;
const V1_SCALE_FACTOR: f32 = 1.0;
