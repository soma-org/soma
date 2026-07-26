# Soma <-> Base Sepolia bridge deployment runbook

Bring up the Soma <-> Eth USDC bridge with **Base Sepolia** as the EVM
side. Soma's `BridgeChainId::BaseSepolia` byte is `13`; Base Sepolia's
EVM chain id is `84532`. Keep that distinction straight throughout —
the on-chain wire format uses the 1-byte Soma id, while the JSON-RPC
provider uses the EVM id.

---

## TL;DR (the 30-second version)

1. `foundryup && cd bridge/evm && forge install`
2. Each validator: register their bridge key via `soma validator register-bridge-key --bridge-pubkey ... --http-url ...`, then wait one epoch.
3. Edit `bridge/evm/deploy_configs/84532.json` once with USDC address, limiter cap, and supported source chains. (Repo ships a sane default.)
4. `cargo run --bin bridge-committee-export -- --soma-rpc http://<rpc>:9000 --target-chain-id 13 --output bridge/evm/deploy_configs/84532.json` (MERGES committee fields into the file; re-runnable per committee rotation.)
5. Fund deployer wallet from https://portal.cdp.coinbase.com/products/faucet
6. `cd bridge/evm && forge script script/DeployBridge.s.sol:DeployBridge --rpc-url $BASE_SEPOLIA_RPC --private-key $DEPLOYER_PK --broadcast --verify --etherscan-api-key $BASESCAN_KEY` (the script auto-reads `deploy_configs/84532.json` keyed off `block.chainid`.)
7. Materialize per-validator configs from `bridge-node/configs/base-sepolia.toml.template` with the four deployed addresses substituted in.
8. Fund every validator's operator wallet with ~0.5 ETH on Base Sepolia.
9. On each validator host: `bridge-node --config /etc/soma/bridge.toml` (under systemd / supervisor).
10. Smoke-test: USDC deposit on Base Sepolia, then a Soma `BridgeWithdraw` round-trip.

---

## Prerequisites

- A Soma chain running with `N >= 4` validators, all reachable via gRPC + the validator-to-validator HTTP port.
- Each validator host is SSH-accessible by the operator running this runbook.
- Foundry installed (`curl -L https://foundry.paradigm.xyz | bash && foundryup`).
- Rust toolchain matching `rust-toolchain.toml` at the repo root.
- `jq`, `curl`, and `cast` on the operator workstation.
- Funded Base Sepolia wallets — see [Funding wallets](#funding-wallets).
- A Base Sepolia JSON-RPC endpoint: Alchemy (`https://base-sepolia.g.alchemy.com/v2/<KEY>`), Ankr (`https://rpc.ankr.com/base_sepolia`), or the public endpoint `https://sepolia.base.org`. Production-grade deployments configure 2-3 providers and let the bridge node's RPC quorum tolerate one failure.
- A BaseScan API key from https://basescan.org/myapikey (only needed for `--verify`).

Export the constants you'll use throughout this runbook into your shell once, up front:

```sh
export BASE_SEPOLIA_RPC=https://base-sepolia.g.alchemy.com/v2/<KEY>
export BASESCAN_KEY=<key>
export DEPLOYER_PK=0x<deployer-private-key>
export SOMA_RPC=http://<soma-fullnode-host>:9000
```

---

## Architecture overview

```
                  +-----------------------------+
                  |        Soma chain           |
                  |  (N validators, BFT cons.)  |
                  |   BridgeState lives in      |
                  |     SystemState object      |
                  +--------------+--------------+
                                 |
                gRPC + checkpoint sub | per-validator
                                 |
        +-----------+------------+------------+-----------+
        |           |                         |           |
        v           v                         v           v
  +-----------+-----------+               +-----------+-----------+
  | bridge-node #1        | ... (peer-broadcast sig aggregator) ..| bridge-node #N
  | eth_syncer, watchdog, |                                       |
  | http_server (sig API),|     <-- HTTP REST signature exchange  |
  | outbound_relayer      |                                       |
  +-----------+-----------+               +-----------+-----------+
              |                                       |
              | finalized-block polling +             |
              | release-tx submission                 |
              v                                       v
        +---------------------------------------------------+
        |              Base Sepolia (EVM 84532)             |
        |   SomaBridge proxy   <--owns--   BridgeVault      |
        |        |    \                       (USDC ERC20)  |
        |        |     \--owns--  BridgeLimiter             |
        |        v                                          |
        |   BridgeCommittee proxy (UUPS, holds N members,   |
        |     2f+1 stake threshold, 10000 BPS total)        |
        +---------------------------------------------------+
```

The watchdog on each bridge node continuously reads (a) the Eth-side
`SomaBridge` USDC balance, (b) the Eth-side `next_withdrawal_nonce`,
and (c) the Soma-side `BridgeState` "total locked" mirror, and
asserts the conservation invariant `eth_locked == soma_minted +
in_flight_tolerance`. Per-poll violations are alert-only; a sustained
violation past `watchdog.failure_threshold` consecutive polls auto-emits
an `EmergencyPause` action through the same signing pipeline as a
normal release.

---

## Pre-deployment validation

Before any deploy or upgrade, validate that every upgradeable bridge
contract is upgrade-safe. The OpenZeppelin Foundry Upgrades plugin
statically checks: storage layout (no reorders / no holes / no retyped
slots), UUPS shape (impl exports `_authorizeUpgrade`, no `selfdestruct`,
no raw `delegatecall`), and initializer wiring. A storage-layout
regression in an impl will brick every existing proxy at upgrade time —
this check is the only thing standing between a typo and a permanent
fund-locking event.

Run:

```sh
cd bridge/evm
forge clean
forge test --force --match-path test/UpgradeValidationTest.t.sol -vvv
```

Expected: 3 `[PASS]` lines, one each for `BridgeCommittee`, `BridgeLimiter`,
and `SomaBridge`.

Failures here MUST block the deploy. The most likely cause is a
state-variable reorder in an impl — fix the order to match the previous
version's layout, **NOT** the validation test.

### One-time machine setup

The OZ plugin shells out to a Node.js binary (`@openzeppelin/upgrades-core`)
via `ffi` to do storage-layout diffing. On a fresh machine `npx` will
auto-install it the first time the test runs (one-time, then cached);
subsequent runs are offline. If the auto-install is blocked by your
network policy, pre-install with:

```sh
npm install --global @openzeppelin/upgrades-core
```

The `forge clean && --force` is intentional — the plugin needs a *full*
compilation in `out/build-info/`, not the partial one Foundry produces
on incremental rebuilds.

---

## Phase 1: Validators register their bridge keys

Each validator must add their secp256k1 bridge pubkey to `BridgeState.bridge_committee` **before** the export step. This is a per-validator on-chain action, not something the operator does centrally.

On each validator host (key generation):

```sh
# Generate the validator's ECDSA bridge key (Secp256k1).
soma keytool generate ecdsa --key-scheme secp256k1 \
    --output bridge.key

# Print the 33-byte compressed pubkey hex for the next step.
soma keytool show bridge.key --pubkey-only
```

Submit the registration transaction. The signer is the validator's Soma
address; the executor verifies the signer is in the current active
validator set before storing the registration. See
`BridgeRegisterBridgeKeyArgs` in `types/src/transaction.rs:564`.

```sh
soma validator register-bridge-key \
    --bridge-pubkey 0x<33-byte-compressed-pubkey-hex> \
    --http-url https://bridge-1.<validator-domain>:9191
```

The `--http-url` value MUST be reachable by peer bridge nodes — it's
where the `http_server` listens for `/sign/...` requests during sig
aggregation. Use a real DNS name and a TLS reverse-proxy in production.

> **Note:** As of PR M/N being in flight, the `soma validator register-bridge-key` subcommand may not yet exist in `cli/src/commands/validator.rs`. If so, submit the `BridgeRegisterBridgeKey` transaction directly via the SDK or by patching the validator command. The on-chain transaction kind is stable (`TransactionKind::BridgeRegisterBridgeKey(BridgeRegisterBridgeKeyArgs)`).

### Wait for an epoch boundary

Bridge registrations take effect at the next `ChangeEpoch`. Watch:

```sh
soma tx <register-tx-digest>     # confirm execution
# Then wait for at least one epoch to pass.
```

### Verification

```sh
# Dry-run the committee export to stdout (no merge) and inspect:
cargo run --bin bridge-committee-export -- \
    --soma-rpc $SOMA_RPC --target-chain-id 13 \
    > /tmp/committee-preview.json 2>/tmp/committee.err

jq '.committeeMembers | length' /tmp/committee-preview.json
# Expect N.
jq '.committeeStake' /tmp/committee-preview.json
# Expect each validator's BPS share, summing to 10000.
```

If `committeeMembers.length` is less than your validator count, one or
more registrations didn't land — re-check each validator's submitted
tx and rerun after the next epoch.

---

## Phase 2: Edit deploy_configs + export the committee

Deployment config lives in `bridge/evm/deploy_configs/<EVM-CHAIN-ID>.json`
(checked into the repo). For Base Sepolia the file is
`bridge/evm/deploy_configs/84532.json`. The Foundry script picks the
file up automatically — there are no env vars to set anymore.

The file has two kinds of fields:

| Field                  | Owned by   | Source                                      |
|------------------------|------------|---------------------------------------------|
| `committeeMembers`     | export tool| `bridge-committee-export` populates         |
| `committeeStake`       | export tool| `bridge-committee-export` populates         |
| `somaCommitteeDigest`  | export tool| `bridge-committee-export` populates         |
| `ethChainId`           | operator   | edit once (BridgeChainId byte)              |
| `usdcAddress`          | operator   | edit once (Circle USDC on target chain)     |
| `limiterTotalLimit`    | operator   | edit per policy (string-encoded uint64)     |
| `supportedSomaChains`  | operator   | edit once (`BridgeChainId` bytes)           |

Edit the operator-owned fields once (the file ships with sane defaults
for Base Sepolia). Then run the export tool with `--output` pointing
at the SAME file — it will MERGE the committee fields into the existing
JSON, preserving all your other fields:

```sh
cargo run --bin bridge-committee-export -- \
    --soma-rpc $SOMA_RPC \
    --target-chain-id 13 \
    --output bridge/evm/deploy_configs/84532.json
```

Re-run this command whenever the on-chain committee rotates. Your
`usdcAddress` / `limiterTotalLimit` / etc. stay untouched.

Example `deploy_configs/84532.json` after a successful export:

```json
{
  "committeeMembers": [
    "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "0xcccccccccccccccccccccccccccccccccccccccc",
    "0xdddddddddddddddddddddddddddddddddddddddd"
  ],
  "committeeStake": [2500, 2500, 2500, 2500],
  "ethChainId": 13,
  "usdcAddress": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
  "limiterTotalLimit": "1000000000000",
  "supportedSomaChains": [2],
  "somaCommitteeDigest": "0x9b1c...e4f0"
}
```

If `--output` is omitted, the export tool writes a full JSON blob to
stdout (with empty operator placeholders); use this for dry-runs and
inspection, never as the deploy file.

### Verification

- `jq '.committeeStake | add' bridge/evm/deploy_configs/84532.json`
  must return `10000`. (`BridgeCommittee.initialize` reverts otherwise.)
- Record `somaCommitteeDigest` in your deployment ledger. The bridge
  nodes recompute it at runtime and refuse to relay if the on-chain
  committee diverges from the one baked into `BridgeCommittee.initialize`.

---

## Phase 3: Fund deployer wallet

1. Open https://portal.cdp.coinbase.com/products/faucet
2. Select **Base Sepolia** in the network dropdown.
3. Paste the deployer EOA address (the one whose private key you'll pass as `$DEPLOYER_PK`).
4. Request the max drip (currently 0.05 ETH per 24h per address; request twice across two CDP accounts if you need more, or use https://www.alchemy.com/faucets/base-sepolia as a secondary source).

### Verification

```sh
cast balance --rpc-url $BASE_SEPOLIA_RPC $(cast wallet address --private-key $DEPLOYER_PK)
# Expect >= 0.05 ETH (5e16 wei). Deployment costs ~0.01-0.02 ETH at
# typical Base Sepolia gas prices.
```

---

## Phase 4: Deploy Eth contracts

Make sure `bridge/evm/deploy_configs/84532.json` is filled in (Phase 2)
and the deployer wallet is funded (Phase 3). Then run the Foundry
script. There are no env vars to pass — the script reads
`deploy_configs/<block.chainid>.json` automatically.

```sh
cd bridge/evm

forge script script/DeployBridge.s.sol:DeployBridge \
    --rpc-url $BASE_SEPOLIA_RPC \
    --private-key $DEPLOYER_PK \
    --broadcast \
    --verify --etherscan-api-key $BASESCAN_KEY
```

Notes on the config fields (see `bridge/evm/deploy_configs/84532.json`):

- `ethChainId: 13` — Soma's `BridgeChainId::BaseSepolia` byte. NOT the
  EVM chain id (`84532`). The script narrows to `uint8` and embeds it
  in `BridgeCommittee.initialize`.
- `usdcAddress` — Circle's Base Sepolia USDC. Stable. Verify at
  https://developers.circle.com/stablecoins/usdc-on-test-networks.
- `limiterTotalLimit: "1000000000000"` — 1,000,000 USDC in micro
  (USDC has 6 decimals). Per-day outbound cap. Tighten or loosen
  per ops policy. **String-encoded** to dodge JSON number-precision
  issues.
- `supportedSomaChains: [2]` — Soma testnet
  (`BridgeChainId::SomaCustom` byte). For multi-source-chain
  deployments pass an array, e.g. `[0, 1, 2]`.

The script's `--broadcast` flag is what actually submits txs;
`--verify` runs Etherscan verification for each contract right after
deployment. If verification flakes (BaseScan can be rate-limited),
re-run it later with `forge verify-contract`.

### CI / integration-test override

For ad-hoc Anvil runs where `block.chainid` doesn't match a deploy_configs
file, set `OVERRIDE_CONFIG_PATH=/abs/path/to/config.json` in the
environment. The script echoes a loud `!! OVERRIDE_CONFIG_PATH is in
effect !!` line whenever it triggers — **do not** rely on this for
production deploys.

### Expected output (final stdout lines)

```
== Soma bridge deployment ==
block.chainid       = 84532
config path         = /.../bridge/evm/deploy_configs/84532.json
ethChainId          = 13
usdcAddress         = 0x036CbD53842c5426634e7929541eC2318f3dCF7e
limiterTotalLimit   = 1000000000000
committee size      = 4
  member 0 0xaaaa...
  stake  0 2500
  ...
supportedSomaChains count = 1
  soma chain 0 2
BRIDGE_COMMITTEE_PROXY= 0x1111...
BRIDGE_VAULT=          0x2222...
BRIDGE_LIMITER_PROXY=  0x3333...
SOMA_BRIDGE_PROXY=     0x4444...
DEPLOYMENT_BLOCK=      12345678
checkmark deployment complete
```

Capture those four addresses + `DEPLOYMENT_BLOCK` — they go straight
into the per-validator config in Phase 5.

### Verification

```sh
# Sanity: BridgeCommittee is initialized and reports your committee size.
cast call $BRIDGE_COMMITTEE_PROXY "committeeSize()(uint256)" \
    --rpc-url $BASE_SEPOLIA_RPC
# Expect: N (your validator count).

# Sanity: BridgeVault is owned by SomaBridge (not the deployer EOA).
cast call $BRIDGE_VAULT "owner()(address)" --rpc-url $BASE_SEPOLIA_RPC
# Expect: $SOMA_BRIDGE_PROXY

cast call $BRIDGE_LIMITER_PROXY "owner()(address)" --rpc-url $BASE_SEPOLIA_RPC
# Expect: $SOMA_BRIDGE_PROXY
```

If either `owner()` returns the deployer address, the
`vault.transferOwnership` / `limiter.transferOwnership` calls at the
end of the Foundry script silently no-op'd — STOP, do not start the
bridge nodes; the deployer EOA could front-run release calls.

Pin the deployed addresses into your operator ledger:

```sh
# Pull the digest from the deploy_configs file you just deployed against.
SOMA_COMMITTEE_DIGEST=$(jq -r .somaCommitteeDigest bridge/evm/deploy_configs/84532.json)

cat <<EOF > deployment-base-sepolia.env
SOMA_COMMITTEE_DIGEST=$SOMA_COMMITTEE_DIGEST
BRIDGE_COMMITTEE_PROXY=0x1111...
BRIDGE_VAULT=0x2222...
BRIDGE_LIMITER_PROXY=0x3333...
SOMA_BRIDGE_PROXY=0x4444...
DEPLOYMENT_BLOCK=12345678
EOF
```

---

## Phase 5: Build per-validator configs

A template lives (or will live; PR N) at
`bridge-node/configs/base-sepolia.toml.template`. Substitute the
deployed addresses + each validator's bridge-key path and operator
wallet hex.

Template fields (cross-reference `bridge-node/src/config.rs`):

```toml
# bridge-node/configs/base-sepolia.toml.template
bridge_key_path = "{{BRIDGE_KEY_PATH}}"
eth_rpc_urls = ["{{ETH_RPC_PRIMARY}}", "{{ETH_RPC_SECONDARY}}"]
bridge_contract_address = "{{SOMA_BRIDGE_PROXY}}"
soma_rpc_url = "{{SOMA_RPC}}"
http_listen_address = "0.0.0.0:9191"
eth_chain_id = 84532          # EVM chain id, NOT the BridgeChainId byte.
eth_poll_interval_ms = 3000   # Base produces ~2s blocks; poll faster than Sepolia.
eth_start_block_fallback = {{DEPLOYMENT_BLOCK}}
wal_path = "/var/lib/soma/bridge-wal"

[watchdog]
usdc_contract_address = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
eth_bridge_contract_address = "{{SOMA_BRIDGE_PROXY}}"
poll_interval_ms = 5000
failure_threshold = 6
in_flight_tolerance_micro = 1000000000   # 1k USDC

[outbound_relayer]
bridge_contract_address = "{{SOMA_BRIDGE_PROXY}}"
operator_private_key_hex = "{{OPERATOR_PK_HEX}}"
poll_interval_ms = 10000
```

Materialize per-validator with a shell loop:

```sh
set -a
source deployment-base-sepolia.env
set +a

for V in val-1 val-2 val-3 val-4; do
    sed -e "s|{{BRIDGE_KEY_PATH}}|/etc/soma/bridge.key|g" \
        -e "s|{{ETH_RPC_PRIMARY}}|${BASE_SEPOLIA_RPC}|g" \
        -e "s|{{ETH_RPC_SECONDARY}}|https://rpc.ankr.com/base_sepolia|g" \
        -e "s|{{SOMA_BRIDGE_PROXY}}|${SOMA_BRIDGE_PROXY}|g" \
        -e "s|{{SOMA_RPC}}|http://soma-${V}.internal:9000|g" \
        -e "s|{{DEPLOYMENT_BLOCK}}|${DEPLOYMENT_BLOCK}|g" \
        -e "s|{{OPERATOR_PK_HEX}}|$(cat secrets/${V}-op.key)|g" \
        bridge-node/configs/base-sepolia.toml.template \
        > out/${V}.toml
done
```

### Verification

```sh
for V in val-1 val-2 val-3 val-4; do
    cargo run --bin bridge-node -- --config out/${V}.toml --validate-only
done
# Expect: "config validated" for each. The validator binary calls
# BridgeNodeConfig::validate() which enforces non-empty eth_rpc_urls,
# non-empty bridge_contract_address, etc.
```

If `--validate-only` isn't a flag yet, just run `bridge-node --config
... --dry-run` or skim each TOML by eye against `config.rs`.

---

## Phase 6: Fund operator wallets

Each validator's bridge node has an operator EOA whose private key sits
in `outbound_relayer.operator_private_key_hex`. That EOA pays gas for
every Eth-side release tx (USDC transfer out of `BridgeVault`).

Per-tx gas cost on Base Sepolia ranges 80k-150k gas at ~1 gwei base fee
= ~1.5e-4 ETH. **0.5 ETH covers ~3000 releases per validator** with
healthy headroom for gas-price spikes.

1. Coinbase Developer Platform faucet: https://portal.cdp.coinbase.com/products/faucet (0.05 ETH/24h/address).
2. Alchemy faucet: https://www.alchemy.com/faucets/base-sepolia.
3. For 0.5 ETH × N validators, request from multiple CDP accounts or use the QuickNode multi-faucet sweep over a few days.

### Verification

```sh
for OP_ADDR in 0x... 0x... 0x... 0x...; do
    cast balance --rpc-url $BASE_SEPOLIA_RPC $OP_ADDR
done
# Expect each >= 5e17 wei (0.5 ETH).
```

---

## Phase 7: Launch bridge nodes

On each validator host, copy the materialized TOML + bridge key into
place and start the daemon under systemd (or your supervisor of
choice).

```sh
# On each validator host:
sudo install -d -o soma -g soma -m 0750 /etc/soma /var/lib/soma
sudo install -o soma -g soma -m 0640 out/val-1.toml /etc/soma/bridge.toml
sudo install -o soma -g soma -m 0400 secrets/val-1-bridge.key /etc/soma/bridge.key

sudo systemctl start soma-bridge
sudo journalctl -u soma-bridge -f
```

Watch the startup logs for the expected sequence:

```
[INFO bridge_node::node] loading config from /etc/soma/bridge.toml
[INFO bridge_node::node] config validated
[INFO bridge_node::eth_syncer] starting eth_syncer, chain_id=84532, start_block=12345678
[INFO bridge_node::soma_syncer] starting soma_syncer
[INFO bridge_node::http_server] HTTP sig API listening on 0.0.0.0:9191
[INFO bridge_node::watchdog] watchdog started, poll_interval=5s, failure_threshold=6
[INFO bridge_node::outbound_relayer] outbound relayer started, scan_window=1024
```

### Verification

Per validator:

```sh
# 1. HTTP ping reachable from peers.
curl -fsS https://bridge-1.<validator-domain>:9191/ping
# Expect: 200 OK with a JSON envelope (Sui parity).

# 2. eth_syncer cursor advancing.
curl -fsS https://bridge-1.<validator-domain>:9191/metrics | grep eth_last_seen_block
# Expect the value to increment every ~3s.

# 3. Soma syncer caught up to chain head.
curl -fsS https://bridge-1.<validator-domain>:9191/metrics | grep soma_checkpoint_cursor
# Expect within a few of the live `soma tx checkpoint` output.

# 4. Watchdog reading both observables.
curl -fsS https://bridge-1.<validator-domain>:9191/metrics | grep watchdog
# Expect: watchdog_eth_balance_micro, watchdog_soma_minted_micro,
# watchdog_consecutive_violations (0).
```

Repeat across all `N` validators. Don't proceed until every node
reports `watchdog_consecutive_violations=0` and the eth cursor is
moving on every node.

---

## Phase 8: Smoke test

### Inbound (Base Sepolia -> Soma)

1. Acquire Base Sepolia USDC from Circle's faucet: https://faucet.circle.com (select Base Sepolia, paste your EOA).
2. Approve the bridge proxy to pull USDC:

   ```sh
   cast send 0x036CbD53842c5426634e7929541eC2318f3dCF7e \
       "approve(address,uint256)" $SOMA_BRIDGE_PROXY 1000000 \
       --rpc-url $BASE_SEPOLIA_RPC --private-key $USER_PK
   ```

3. Deposit 1 USDC into the bridge, targeting your Soma recipient:

   ```sh
   cast send $SOMA_BRIDGE_PROXY \
       "bridgeERC20(uint8,address,uint256,bytes)" \
       2 0x036CbD53842c5426634e7929541eC2318f3dCF7e 1000000 0x<32-byte-soma-recipient> \
       --rpc-url $BASE_SEPOLIA_RPC --private-key $USER_PK
   ```

4. Wait for Base Sepolia finalization (~13 blocks, ~30s under
   stable conditions). The bridge nodes' `eth_syncer` only consumes
   finalized blocks.
5. Verify the Soma-side `BridgeRecord` materialized:

   ```sh
   soma object list 0x<recipient> | grep BridgeRecord
   soma object get <bridge-record-id>
   ```

Expected timing end-to-end: ~60-90 seconds from `bridgeERC20` tx
inclusion to Soma-side mint, depending on Base finalization cadence.

### Outbound (Soma -> Base Sepolia)

1. Initiate a withdrawal on Soma:

   ```sh
   soma bridge withdraw \
       --amount 1000000 \
       --eth-recipient 0x<recipient-on-base> \
       --target-chain 13
   ```

2. Wait for committee sig aggregation (~one Soma checkpoint, then
   one round of HTTP fetch-and-sign). Watch on each validator:

   ```sh
   sudo journalctl -u soma-bridge -f | grep -E 'PendingWithdrawal|aggregator'
   ```

   You should see each peer's bridge node fetch the others' sigs,
   then one node (usually the lowest-stake-index member) attach the
   final cert via `BridgeAttachWithdrawalSignatures`.

3. Watch the outbound relayer logs for the Eth submission:

   ```sh
   sudo journalctl -u soma-bridge -f | grep outbound_relayer
   ```

   Expected line: `submitted release tx <hash> for nonce <n>`.

4. Verify USDC received on Base Sepolia:

   ```sh
   cast call 0x036CbD53842c5426634e7929541eC2318f3dCF7e \
       "balanceOf(address)(uint256)" 0x<recipient-on-base> \
       --rpc-url $BASE_SEPOLIA_RPC
   ```

### Watchdog validation

1. **Tolerated divergence** (alert-only). Send a deposit but kill the
   `eth_syncer` on one validator before it observes. That node's
   `watchdog_consecutive_violations` should rise to 1-2 then reset
   once the syncer catches up. Confirm via `journalctl`:

   ```
   [WARN bridge_node::watchdog] tolerated divergence: eth=N soma=M diff=...
   ```

2. **Auto-pause trip**. Force a sustained violation by stopping the
   `outbound_relayer` on a quorum of nodes WHILE Eth-side releases
   are pending. After `failure_threshold` (default 6) consecutive
   polls (~30s), the watchdog will queue an `EmergencyPause` action
   through the same signing pipeline as a normal release.
   You'll see:

   ```
   [ERROR bridge_node::watchdog] auto-pause threshold reached, queueing EmergencyPause nonce=<n>
   ```

   Verify on Eth: `cast call $SOMA_BRIDGE_PROXY "paused()(bool)" --rpc-url $BASE_SEPOLIA_RPC` returns `true`.

3. **Manual unpause**. Operators sign a quorum `Unpause` message
   off-chain (each validator approves via their `approved_governance_actions` list — see Troubleshooting on governance whitelist),
   then any party submits the cert via `cast send $SOMA_BRIDGE_PROXY "executeEmergencyOpWithSignatures(bytes,bytes[])" ...`.
   `paused()` returns `false` afterward.

---

## Troubleshooting

| Symptom | Diagnosis & fix |
|---|---|
| `DeployBridge: committeeStake must sum to 10000 BPS` revert | `deploy_configs/<chain>.json` was hand-edited or the export merge corrupted it. Rerun `bridge-committee-export --output deploy_configs/<chain>.json`; the tool normalizes to exactly 10000 BPS. |
| `DeployBridge: committeeMembers is empty — run bridge-committee-export ...` revert | You haven't run `bridge-committee-export --output deploy_configs/<chain>.json` yet (or the file ships with empty `committeeMembers` and you forgot). Run it. |
| `DeployBridge: ethChainId exceeds uint8` revert | You set `ethChainId` to `84532` (EVM chain id) instead of `13` (`BridgeChainId::BaseSepolia` byte) in `deploy_configs/84532.json`. |
| Bridge node startup fails: `Bridge key file not found` | `bridge_key_path` in the TOML doesn't match where you copied the key on the host. Check `ls -l /etc/soma/bridge.key` and that the daemon user can read it. |
| Bridge node can't reach peers (sig aggregation hangs) | The `http_url` registered on-chain in Phase 1 isn't reachable from peer nodes' egress. Open the port in your firewall/security group; verify with `curl https://bridge-N:9191/ping` from each peer. |
| Inbound deposit observed (event seen in logs) but never signed | One of: (a) the deposit's source chain id isn't in `SUPPORTED_SOMA_CHAINS`, (b) the deposit amount blows the `BridgeLimiter` daily cap, (c) the validator's operator hasn't whitelisted the action shape in `approved_governance_actions`. Inbound token transfers do NOT require governance whitelisting — they're server-verified — so (c) only applies to governance txs. |
| Outbound release reverts on Eth with `BridgeCommittee: insufficient stake` | Committee mismatch. Compare `soma_committee_digest` in your ledger vs. what bridge nodes log at startup. If they differ, the on-chain Soma committee has rotated since deployment but the Eth `BridgeCommittee` hasn't been UUPS-upgraded — see [Recovery](#recovery-procedures). |
| Watchdog log: `EthVaultBalance: read failed` | All configured Eth RPC URLs flapped at once. Add a third provider to `eth_rpc_urls`; the bridge node's RPC fan-out is happy with three providers and one failing. |
| Watchdog log: `tolerated divergence` then auto-pause | The conservation invariant exceeded `in_flight_tolerance_micro` for `failure_threshold` consecutive polls. Inspect both observables in metrics; usually this means either (a) a stuck release tx (Eth side too slow), or (b) the relayer is double-submitting. Tighten the tolerance only after you've ruled out a real loss. |
| `forge script` succeeds but `--verify` reports timeout | BaseScan is rate-limited or the API key is wrong. Rerun with `forge verify-contract --etherscan-api-key $BASESCAN_KEY <addr> <contract>` per contract. |
| Outbound relayer logs `wallet underfunded, skipping nonce <n>` | Operator EOA ran out of Base Sepolia ETH. Refund from the faucet; the relayer retries from WAL state, so no records are lost. |

---

## Recovery procedures

### Restarting a bridge node

```sh
sudo systemctl restart soma-bridge
```

The WAL at `wal_path` persists pending actions, the per-Eth-contract
block cursor, and the Soma checkpoint cursor. On restart the node
resumes from those cursors instead of re-scanning from genesis. A
clean restart should reach steady state inside one poll interval
(~5s default).

### Adding a new validator post-launch

1. New validator joins the Soma validator set (normal `AddValidator` flow).
2. New validator registers their bridge pubkey: `soma validator register-bridge-key ...` (see Phase 1).
3. Wait one Soma epoch.
4. Operator re-exports the committee: `bridge-committee-export -- --soma-rpc $SOMA_RPC --target-chain-id 13 ...`.
5. Operator UUPS-upgrades `BridgeCommittee` with new `initialize` args carrying the updated member list. The committee is a UUPS proxy precisely so this is doable without redeploying `SomaBridge` / `BridgeVault` / `BridgeLimiter`.
6. Watch every bridge node's startup log to confirm they pick up the new committee digest. If a node logs `committee digest mismatch`, its config is stale or its RPC is reading a forked Eth state.

### Rotating committee membership

Same flow as above. UUPS upgrade of `BridgeCommittee` with new
`initialize` args (members + stakes summing to 10000). The
`bridge-committee-export` tool's digest is your invariant: if the
post-rotation digest matches what the chain reports, every node is
on the new committee.

### Emergency pause via manual operator

For situations where the watchdog hasn't tripped but the operator has
out-of-band evidence of a problem (e.g. a Circle USDC contract
incident on Base Sepolia):

1. Operator constructs `BridgeAction::EmergencyPause { nonce: <current_emergency_op_seq> }` and circulates it to every validator out-of-band.
2. Each validator adds the action shape to `approved_governance_actions` in their bridge-node config and restarts.
3. Operator hits each bridge node's `/sign/emergency-pause/<nonce>` HTTP endpoint to collect signatures.
4. With 2f+1 sigs collected, submit:

   ```sh
   cast send $SOMA_BRIDGE_PROXY \
       "executeEmergencyOpWithSignatures(bytes,bytes[])" \
       <msg-bytes> <sig-array> \
       --rpc-url $BASE_SEPOLIA_RPC --private-key $OPERATOR_PK
   ```

5. Verify `paused()(bool)` returns `true`.

Unpause uses the same flow with the `Unpause` action shape — keep
both pre-approved in `approved_governance_actions` ahead of time so
you're not racing whitelist deploys during an incident.

---

## Reference: addresses + endpoints

| Resource | Value |
|---|---|
| Base Sepolia chain id (EVM) | `84532` |
| BridgeChainId byte (Soma wire format) | `13` (`BridgeChainId::BaseSepolia`) |
| Circle USDC on Base Sepolia | `0x036CbD53842c5426634e7929541eC2318f3dCF7e` |
| Coinbase Developer faucet | https://portal.cdp.coinbase.com/products/faucet |
| Alchemy faucet (Base Sepolia ETH) | https://www.alchemy.com/faucets/base-sepolia |
| Circle USDC faucet | https://faucet.circle.com |
| BaseScan | https://sepolia.basescan.org |
| BaseScan API keys | https://basescan.org/myapikey |
| Public Base RPC | https://sepolia.base.org |
| Alchemy template | `https://base-sepolia.g.alchemy.com/v2/<KEY>` |
| Ankr template | `https://rpc.ankr.com/base_sepolia` |
| QuickNode template | `https://<endpoint>.base-sepolia.quiknode.pro/<KEY>/` |
| Block explorer (txs) | `https://sepolia.basescan.org/tx/<hash>` |
| Block explorer (addrs) | `https://sepolia.basescan.org/address/<addr>` |
| Soma `BridgeChainId::SomaCustom` byte | `2` |
| Default bridge-node HTTP port | `9191` |
| Default watchdog poll interval | `5000ms` |
| Default watchdog failure threshold | `6` polls (~30s) |
| Default outbound relayer poll interval | `10000ms` |
| Default outbound scan window | `1024` nonces |

### Code references

| Topic | File |
|---|---|
| Foundry deploy script + JSON schema | [`bridge/evm/script/DeployBridge.s.sol`](evm/script/DeployBridge.s.sol) |
| Per-chain deploy config (Base Sepolia) | [`bridge/evm/deploy_configs/84532.json`](evm/deploy_configs/84532.json) |
| Committee export tool CLI | [`bridge-node/src/bin/bridge_committee_export.rs`](../bridge-node/src/bin/bridge_committee_export.rs) |
| Bridge node config struct | [`bridge-node/src/config.rs`](../bridge-node/src/config.rs) |
| Watchdog behavior | [`bridge-node/src/watchdog.rs`](../bridge-node/src/watchdog.rs) |
| `BridgeRegisterBridgeKey` tx | [`types/src/transaction.rs:564`](../../types/src/transaction.rs) |
| `BridgeChainId` enum | [`types/src/bridge.rs:39`](../../types/src/bridge.rs) |
| Wire format (must match Sol) | [`types/src/bridge.rs::encode_bridge_message`](../../types/src/bridge.rs) <-> [`bridge/evm/contracts/utils/BridgeMessage.sol`](evm/contracts/utils/BridgeMessage.sol) |
