from dataclasses import dataclass

from soma_models.config import (
    V1_EMBEDDING_DIM,
    V1_NUM_LAYERS,
    V1_PWFF_HIDDEN_DIM,
    V1_NUM_HEADS,
    V1_VOCAB_SIZE,
    V1_MAX_WAVELENGTH,
    V1_SCALE_FACTOR,
    V1_SIG_REG_T_MAX,
    V1_SIG_REG_POINTS,
    V1_SIG_REG_SLICES,
)


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
