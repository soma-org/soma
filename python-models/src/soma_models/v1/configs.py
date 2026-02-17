from dataclasses import dataclass

V1_PAD_TOKEN_ID = 256
V1_EOS_TOKEN_ID = 257

V1_EMBEDDING_DIM = 2048
V1_NUM_HEADS = 8
V1_NUM_LAYERS = 32
V1_MAX_SEQ_LEN = 8192
V1_PWFF_HIDDEN_DIM = V1_EMBEDDING_DIM * 4
V1_MAX_WAVELENGTH = 10_000.0
V1_SCALE_FACTOR = 1.0
V1_VOCAB_SIZE = 256 + 8

V1_SIG_REG_T_MAX = 3.0
V1_SIG_REG_SLICES = 1024
V1_SIG_REG_POINTS = 17
V1_SIG_REG_COEFFICIENT = 0.02

V1_BATCH_SIZE = 32


@dataclass
class ModelConfig:
    dropout_rate: float
    embedding_dim: int = V1_EMBEDDING_DIM
    pwff_hidden_dim: int = V1_PWFF_HIDDEN_DIM
    num_layers: int = V1_NUM_LAYERS
    num_heads: int = V1_NUM_HEADS
    vocab_size: int = V1_VOCAB_SIZE
    max_wavelength: float = V1_MAX_WAVELENGTH
    scale_factor: float = V1_SCALE_FACTOR


@dataclass
class EncoderConfig:
    dropout_rate: float
    embedding_dim: int = V1_EMBEDDING_DIM
    pwff_hidden_dim: int = V1_PWFF_HIDDEN_DIM
    num_layers: int = V1_NUM_LAYERS
    num_heads: int = V1_NUM_HEADS
    max_wavelength: float = V1_MAX_WAVELENGTH
    scale_factor: float = V1_SCALE_FACTOR


@dataclass
class LayerConfig:
    dropout_rate: float
    embedding_dim: int = V1_EMBEDDING_DIM
    pwff_hidden_dim: int = V1_PWFF_HIDDEN_DIM
    num_heads: int = V1_NUM_HEADS
    max_wavelength: float = V1_MAX_WAVELENGTH
    scale_factor: float = V1_SCALE_FACTOR


@dataclass
class PositionWiseFeedForwardConfig:
    dropout_rate: float
    embedding_dim: int = V1_EMBEDDING_DIM
    pwff_hidden_dim: int = V1_PWFF_HIDDEN_DIM


@dataclass
class SIGRegConfig:
    t_max: float = V1_SIG_REG_T_MAX
    points: int = V1_SIG_REG_POINTS
    slices: int = V1_SIG_REG_SLICES
    coefficient: float = V1_SIG_REG_COEFFICIENT
