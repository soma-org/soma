# soma-arrgen

Deterministic array generation for the SOMA network. Produces identical arrays in both Rust and Python, ensuring numerical reproducibility across the on-chain runtime and Python training environments.

Used to generate deterministic model weights, inputs, and noise (e.g., SIGReg) so that the same seed always produces the same output regardless of language or platform.

## Install

```bash
uv add soma-arrgen
```

Or with pip:

```bash
pip install soma-arrgen
```

## Usage

```python
from arrgen import normal_array, uniform_array, constant_array

# Deterministic normal distribution
arr = normal_array(seed=42, shape=[2048, 256], mean=0.0, std_dev=1.0)

# Deterministic uniform distribution
arr = uniform_array(seed=42, shape=[32, 8192], min=0.0, max=1.0)

# Constant-filled array
arr = constant_array(shape=[64, 64], value=0.5)
```

## Development

Reinstall after making changes (run from the workspace root):

```bash
uv sync --reinstall-package soma-arrgen
```

Run Python tests:

```bash
uv run pytest arrgen
```

Run Rust tests:

```bash
cargo test -p arrgen
```