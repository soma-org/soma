# Soma Python Library

## Development Guide

### Setup

#### Install maturin

```
cargo binstall maturin
```

#### Setup virtual env
```bash
uv venv
```

The path should be `soma-python/.venv`

#### Add required python deps
```bash
uv pip install pip
uv pip install maturin
```

#### Build
```bash
maturin develop
```