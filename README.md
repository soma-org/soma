```
RUSTFLAGS="--cfg msim" RUST_BACKTRACE=1 cargo test -p e2e-tests --test reconfiguration_tests test_reconfig_with_committee_change_basic

RUSTFLAGS="--cfg msim" RUST_BACKTRACE=1 cargo test -p authority --lib -- state::authority_tests --test test_conflicting_transactions

RUST_BACKTRACE=1 cargo test -p authority --lib -- pay_coin_tests
```