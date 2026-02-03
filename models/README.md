# Probes

The probe is a generic pre-norm transformer. The hyperparameters were selected to keep the serialized parameter account around 200MB and memory consumption low since probe weights are shared over internet and ensembled. 

## Hyperparameters

### V1
```
EMBEDDING_DIM = 1024
VOCAB_SIZE = 256
NUM_HEADS = 8
NUM_LAYERS = 4
MAX_SEQ_LEN = 256
PWFF_HIDDEN_DIM = EMBEDDING_DIM * 4
MAX_WAVELENGTH = 100_000.0
```

Given the 200MB budget, V1 optimizes fewer layers but a larger embedding dimension and corresponding feed forward dimension.

A shallow and wide architecture is adopted for a few different reasons:

1. The probe is meant to behave similar to a linear probe, such that the bulk of enrichment should happen prior to the probe model. Flattening the model helps with convergence with less data and limits the complexity of what the probe weights can store.

2. From reviewing various 2025 small model architectures it seems that a common trend is to reduce layer count, expand FF size, and increase number of attention heads to improve performance.
