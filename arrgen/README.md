# Array Generation (arrgen)
Generates deterministic and identical arrays in both rust and python environments.

Used to test probes implementations. The typical flow is to generate model weights and inputs deterministically using this library and then expect a specific output.

## Reinstall
Run this at the workspace root after making changes to the 'arrgen' library

```
uv sync --reinstall-package arrgen
```

## Testing in Python

```
uv run pytest arrgen
```

## Testing in Rust
```
```
