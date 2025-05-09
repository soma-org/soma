# Soma

```
RUSTFLAGS="--cfg msim" RUST_BACKTRACE=full cargo test -p e2e-tests --test encoder_committee_tests
```

```
nix develop --experimental-features 'nix-command flakes'
```

## Node

```
RUSTFLAGS="--cfg msim" RUST_BACKTRACE=1 cargo test -p e2e-tests --test reconfiguration_tests test_reconfig_with_committee_change_basic

RUSTFLAGS="--cfg msim" RUST_BACKTRACE=1 cargo test -p authority --lib -- state::authority_tests --test test_conflicting_transactions

RUST_BACKTRACE=1 cargo test -p authority --lib -- pay_coin_tests
```

## Shard

```bash
tree -a -I target -I .git
```

## Checks

```bash
cargo t # Runs nextest and checks coverage
cargo check
cargo clippy
cargo doc
```
