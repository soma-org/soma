# Bridge-node configs

Per-validator TOML configs for the Soma bridge node. The repo ships one
template — [`base-sepolia.toml.template`](base-sepolia.toml.template) — for
the Phase 1 Base Sepolia ops launch. Operators sed-substitute the
`{{...}}` placeholders to produce a per-validator config, then launch
the bridge node against it.

The template tracks the field order of
[`BridgeNodeConfig`](../src/config.rs); inline comments next to every
field document what it does and what a sensible Phase 1 value is.

## Rendering a config

Single validator:

```sh
sed \
  -e "s|{{VALIDATOR_NAME}}|alice|g" \
  -e "s|{{OPERATOR_PRIVATE_KEY_HEX}}|0xabc123...|g" \
  -e "s|{{ALCHEMY_API_KEY}}|abcd1234|g" \
  -e "s|{{SOMA_BRIDGE_CONTRACT_ADDRESS}}|0xBRIDGE...|g" \
  -e "s|{{USDC_CONTRACT_ADDRESS}}|0x036CbD53842c5426634e7929541eC2318f3dCF7e|g" \
  -e "s|{{SOMA_RPC_URL}}|http://localhost:9000|g" \
  -e "s|{{ETH_DEPLOY_BLOCK}}|12345678|g" \
  configs/base-sepolia.toml.template > configs/alice.toml
```

Many validators in one pass (reads each operator's key from `secrets/`):

```sh
for v in alice bob carol dave; do
  sed \
    -e "s|{{VALIDATOR_NAME}}|$v|g" \
    -e "s|{{OPERATOR_PRIVATE_KEY_HEX}}|$(cat secrets/$v.eth.key)|g" \
    -e "s|{{ALCHEMY_API_KEY}}|$ALCHEMY_API_KEY|g" \
    -e "s|{{SOMA_BRIDGE_CONTRACT_ADDRESS}}|$SOMA_BRIDGE_CONTRACT_ADDRESS|g" \
    -e "s|{{USDC_CONTRACT_ADDRESS}}|$USDC_CONTRACT_ADDRESS|g" \
    -e "s|{{SOMA_RPC_URL}}|http://localhost:9000|g" \
    -e "s|{{ETH_DEPLOY_BLOCK}}|$ETH_DEPLOY_BLOCK|g" \
    configs/base-sepolia.toml.template > configs/$v.toml
done
```

## Substitution table

| Placeholder | Where it comes from |
|---|---|
| `{{VALIDATOR_NAME}}` | Operator's chosen short name. Used in paths (`bridge_key_path`, `wal_path`). Lowercase alphanum. |
| `{{ALCHEMY_API_KEY}}` | Alchemy dashboard → create a Base Sepolia app → copy API key. |
| `{{SOMA_BRIDGE_CONTRACT_ADDRESS}}` | `forge script bridge/evm/script/DeployBridge.s.sol --broadcast ...` stdout — the `SomaBridge proxy:` line. Same value for every validator in the committee. |
| `{{USDC_CONTRACT_ADDRESS}}` | Circle's Base Sepolia USDC: `0x036CbD53842c5426634e7929541eC2318f3dCF7e`. Verify against [Circle's docs](https://developers.circle.com/stablecoins/usdc-on-test-networks) before paste. |
| `{{SOMA_RPC_URL}}` | The validator's own fullnode RPC — usually `http://localhost:9000`. Don't share an RPC across validators. |
| `{{ETH_DEPLOY_BLOCK}}` | `jq '.transactions[0].blockNumber' bridge/evm/broadcast/DeployBridge.s.sol/84532/run-latest.json` (decimal). Avoids re-scanning logs back to genesis on first WAL boot. |
| `{{OPERATOR_PRIVATE_KEY_HEX}}` | Per-validator Eth wallet — see "Provisioning the operator wallet" below. **NOT** the bridge committee key. |

## Provisioning the operator wallet

Each validator needs an Ethereum wallet that pays gas for outbound
release txs on Base Sepolia. This is separate from the bridge committee
key (which signs the inner cert).

1. Generate a fresh secp256k1 keypair — `cast wallet new` (Foundry) or
   your wallet of choice. Store the 32-byte private key hex in a
   per-validator secret (e.g. `secrets/alice.eth.key`, mode 0600).
2. Fund the wallet's Eth address with Base Sepolia ETH for gas. Use the
   [Coinbase Developer Platform faucet](https://portal.cdp.coinbase.com/products/faucet)
   — pick "Base Sepolia" and paste the address. The faucet dispenses
   ~0.1 ETH which is enough for ~1000 release txs at Phase 1 gas prices.
3. (Optional) Top up later via the same faucet or a Base Sepolia bridge
   from Sepolia mainnet.

The Phase 1 launch checklist says one funded operator wallet per
validator — do NOT share an operator wallet across the committee, the
EIP-1559 nonce contention will starve some validators of submission
slots.

## Required Base Sepolia RPC endpoints

The bridge node multi-homes its Eth reads across three providers. The
template ships with this exact list in `eth_rpc_urls` (in this order):

| Provider | URL | Auth | Notes |
|---|---|---|---|
| Alchemy | `https://base-sepolia.g.alchemy.com/v2/<KEY>` | API key | Primary. Paid tier recommended for Phase 1 — free tier rate-limits `eth_getLogs`. |
| Ankr | `https://rpc.ankr.com/base_sepolia` | None (free public) | Fallback. Slower; tighter rate limit. |
| Public Base | `https://sepolia.base.org` | None | Last-resort. Coinbase-hosted. Aggressive rate limit — only used when both above fail. |

Order is load-bearing: the bridge node fans reads across providers but
sends the first attempt to `eth_rpc_urls[0]`. Putting the public RPC
first will get the node rate-limited within minutes.

## Validation

Once the bridge-node main binary lands (currently only
`bridge-committee-export` ships from `bridge-node/Cargo.toml`), validate
the rendered config without starting a real node:

```sh
cargo run --bin bridge-node -- check-config --path configs/alice.toml
```

This runs `BridgeNodeConfig::validate()` from
[`src/config.rs`](../src/config.rs) — verifies the bridge key file
exists, `eth_rpc_urls` is non-empty, `bridge_contract_address` and
`soma_rpc_url` are set. Until the main binary lands, the same validation
runs at bridge-node startup via the node's `BridgeNodeConfig::validate`
call path; a parse-only smoke check is:

```sh
cargo test -p bridge-node config::tests
```

## Common config mistakes

- **Wrong chain id.** `eth_chain_id` is the *EVM* chain id (84532 for
  Base Sepolia), distinct from Soma's wire-format `BridgeChainId` byte
  tag (`BaseSepolia = 13`). The on-chain SomaBridge contract reads this
  back to assert it was deployed against the right chain — a mismatch
  here makes every withdrawal fail signature verification.
- **Unfunded operator wallet.** The bridge node will start, sign certs,
  and try to relay — every `eth_sendRawTransaction` will fail with
  "insufficient funds for gas". Check the operator address's Base
  Sepolia ETH balance before declaring the node up.
- **Shared operator wallet across validators.** Don't. EIP-1559 nonce
  contention will starve some validators of submission slots — the
  symptom is intermittent withdrawal lag with no error in the bridge-
  node logs (the cert is signed and broadcast, the submitter just can't
  land a tx).
- **`bridge_key_path` doesn't exist.** Validation fails fast. Generate
  the bridge key via the standard validator key flow before rendering
  this config.
- **Mismatched `bridge_contract_address` across the committee.** Every
  validator MUST point at the same proxy. A mismatch silently routes
  one validator's reads against the wrong contract, the watchdog will
  trip on stale invariants, and the node will auto-pause.
- **`eth_start_block_fallback = 0`.** The first WAL boot will fan
  `eth_getLogs` requests across millions of blocks back to Base Sepolia
  genesis. Set this to the actual `DeployBridge.s.sol` broadcast block
  — pull it from `broadcast/DeployBridge.s.sol/84532/run-latest.json`.
- **Field order edits without re-checking against `config.rs`.** The
  template tracks struct order in `BridgeNodeConfig` deliberately —
  keep it that way so a `git diff` after a struct change is greppable.
