pub mod modules;
pub mod probe;
pub mod sig_reg;

const V1_EMBEDDING_DIM: usize = 1024;
const V1_NUM_HEADS: usize = 8;
const V1_NUM_LAYERS: usize = 4;
const V1_MAX_SEQ_LEN: usize = 256;
const V1_PWFF_HIDDEN_DIM: usize = V1_EMBEDDING_DIM * 4;

const V1_SIG_REG_T_MAX: f64 = 3.0;
const V1_SIG_REG_SLICES: usize = 1024;
const V1_SIG_REG_POINTS: usize = 17;
