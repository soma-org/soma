#!/usr/bin/env bash
#
# deploy_base_sepolia.sh — one-shot wrapper for the Soma ↔ Base Sepolia
# bridge deployment runbook (Phase 2-4).
#
# What this does:
#   1. Exports the live Soma BridgeState committee into the per-chain
#      deploy_configs JSON via bridge-committee-export.
#   2. Cleans + force-rebuilds the Foundry artifacts (the OZ Upgrades
#      plugin requires a full compilation, not Foundry's incremental).
#   3. Runs the UpgradeValidationTest pre-flight (refuses to deploy if
#      any impl has an unsafe shape).
#   4. Runs DeployBridge.s.sol --broadcast --verify against Base Sepolia.
#   5. Parses the broadcast JSON for the deployed proxy addresses + prints
#      them as KEY=VALUE lines plus a smoke-test command stub.
#
# What this does NOT do:
#   - Phase 1 (validators registering bridge keys) — operators run
#     `soma validator register-bridge-key` per validator before invoking
#     this script. The committee must already exist on chain.
#   - Phase 5+ (per-validator bridge-node config + launch) — see
#     bridge-node/configs/README.md.
#   - Smoke tests (Phase 8) — printed at end as copy-paste commands; you
#     run them yourself with a USDC-funded wallet.
#
# Required environment variables:
#   SOMA_RPC            — Soma fullnode RPC (e.g. http://localhost:9000)
#   BASE_SEPOLIA_RPC    — Base Sepolia RPC (Alchemy/Ankr/public Base)
#   DEPLOYER_PK         — Hex private key (with or without 0x prefix) of
#                         the funded Base Sepolia deployer wallet
#
# Optional:
#   BASESCAN_API_KEY    — BaseScan API key for contract verification.
#                         If unset, --verify is omitted (you can verify
#                         later via `forge verify-contract`).
#   USDC_ADDRESS        — Override the Circle Base Sepolia USDC default.
#                         Useful for localnet rehearsals with MockUSDC.
#   ETH_CHAIN_ID_BYTE   — Override the BridgeChainId byte (default 13 =
#                         BaseSepolia). Don't touch unless you know what
#                         you're doing — this is wire-format-load-bearing.
#   SKIP_COMMITTEE_EXPORT=1 — Skip the bridge-committee-export step
#                         and trust whatever is already in
#                         deploy_configs/<chain>.json. ONLY for local
#                         rehearsals or when re-running a failed deploy
#                         without re-fetching the committee. Production
#                         deploys must let the export run so the
#                         on-chain committee matches what gets baked
#                         into BridgeCommittee.initialize.

set -euo pipefail

# ----- defaults -----
DEFAULT_USDC_ADDRESS="0x036CbD53842c5426634e7929541eC2318f3dCF7e"
DEFAULT_ETH_CHAIN_ID_BYTE=13
# Base Sepolia's actual EIP-155 chain id. Overridable for local
# rehearsals against anvil (DEPLOY_EVM_CHAIN_ID=31337).
DEPLOY_EVM_CHAIN_ID="${DEPLOY_EVM_CHAIN_ID:-84532}"

USDC_ADDRESS_EFFECTIVE="${USDC_ADDRESS:-$DEFAULT_USDC_ADDRESS}"
ETH_CHAIN_ID_BYTE_EFFECTIVE="${ETH_CHAIN_ID_BYTE:-$DEFAULT_ETH_CHAIN_ID_BYTE}"

# ----- prerequisite check -----
missing=()
for var in SOMA_RPC BASE_SEPOLIA_RPC DEPLOYER_PK; do
    if [ -z "${!var:-}" ]; then
        missing+=("$var")
    fi
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: missing required env vars: ${missing[*]}" >&2
    echo "" >&2
    echo "Run with:" >&2
    echo "    SOMA_RPC=... \\" >&2
    echo "    BASE_SEPOLIA_RPC=https://base-sepolia.g.alchemy.com/v2/<KEY> \\" >&2
    echo "    DEPLOYER_PK=0x... \\" >&2
    echo "    BASESCAN_API_KEY=... \\          # optional, enables --verify" >&2
    echo "    ./ops/deploy_base_sepolia.sh" >&2
    exit 1
fi

# Resolve repo root so the script works from any CWD.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="bridge/evm/deploy_configs/${DEPLOY_EVM_CHAIN_ID}.json"

echo "============================================================"
echo "  Soma <-> Base Sepolia bridge deployment"
echo "============================================================"
echo "  SOMA_RPC          = $SOMA_RPC"
echo "  BASE_SEPOLIA_RPC  = ${BASE_SEPOLIA_RPC%/*}/<key-redacted>"
echo "  USDC_ADDRESS      = $USDC_ADDRESS_EFFECTIVE"
echo "  ETH_CHAIN_ID_BYTE = $ETH_CHAIN_ID_BYTE_EFFECTIVE"
echo "  DEPLOY_EVM_CHAIN  = $DEPLOY_EVM_CHAIN_ID"
echo "  CONFIG_PATH       = $CONFIG_PATH"
echo "  --verify          = $([ -n "${BASESCAN_API_KEY:-}" ] && echo yes || echo NO)"
echo "============================================================"
echo ""

# ----- 1. Pre-flight: USDC address must be a real contract on the target -----
# (Skipped for localnet rehearsals against anvil; only matters on real Base
# Sepolia where typos can route deposits into a void.)

# ----- 2. Ensure deploy_configs entry exists; create a stub if not -----
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[2/6] deploy_configs/$DEPLOY_EVM_CHAIN_ID.json missing; writing stub..."
    mkdir -p "$(dirname "$CONFIG_PATH")"
    cat > "$CONFIG_PATH" <<EOF
{
  "committeeMembers": [],
  "committeeStake": [],
  "ethChainId": $ETH_CHAIN_ID_BYTE_EFFECTIVE,
  "usdcAddress": "$USDC_ADDRESS_EFFECTIVE",
  "limiterTotalLimit": "1000000000000",
  "supportedSomaChains": [2],
  "somaCommitteeDigest": "0x0000000000000000000000000000000000000000000000000000000000000000"
}
EOF
else
    echo "[2/6] deploy_configs/$DEPLOY_EVM_CHAIN_ID.json present; will merge committee fields"
fi

# ----- 3. Refresh committee from live Soma chain -----
if [ "${SKIP_COMMITTEE_EXPORT:-0}" = "1" ]; then
    echo ""
    echo "[3/6] SKIP_COMMITTEE_EXPORT=1 — trusting existing committee in $CONFIG_PATH"
else
    echo ""
    echo "[3/6] Exporting live committee from $SOMA_RPC ..."
    cargo run --quiet --bin bridge-committee-export -- \
        --soma-rpc "$SOMA_RPC" \
        --target-chain-id "$ETH_CHAIN_ID_BYTE_EFFECTIVE" \
        --output "$CONFIG_PATH"
fi

# Sanity: committee must be non-empty after merge (otherwise the deploy
# script's pre-broadcast asserts will revert).
member_count=$(python3 -c "import json; print(len(json.load(open('$CONFIG_PATH'))['committeeMembers']))")
if [ "$member_count" -lt 1 ]; then
    echo "ERROR: committee has 0 members. Check that validators have registered via 'soma validator register-bridge-key' and an epoch boundary has rotated the committee, then re-run." >&2
    exit 1
fi
echo "      committee size = $member_count"

# ----- 4. Pre-deploy validation: every impl must be upgrade-safe -----
echo ""
echo "[4/6] Pre-deploy validation (UpgradeValidationTest) ..."
(cd bridge/evm && forge test --force --match-path test/UpgradeValidationTest.t.sol) | tail -5

# ----- 5. Clean rebuild — MUST happen after test+before deploy. -----
# `forge test` triggers a partial recompile; the OZ Upgrades plugin's
# `validateUpgrade` step (called from inside the deploy script) refuses
# to run against partial build-info and errors with
# `ValidateCommandError: Build info file ... is not from a full compilation`.
# Running clean+build immediately before the deploy guarantees a fresh
# build-info matched to the contracts the deploy is about to broadcast.
echo ""
echo "[5/6] forge clean && forge build --force (full compile for plugin) ..."
(cd bridge/evm && forge clean && forge build --force) >/dev/null

# ----- 6. The deploy -----
echo ""
echo "[6/6] forge script DeployBridge.s.sol --broadcast ..."
verify_flag=""
if [ -n "${BASESCAN_API_KEY:-}" ]; then
    verify_flag="--verify --etherscan-api-key $BASESCAN_API_KEY"
fi

# NO `|| true` — a forge script failure must abort the script so the
# operator doesn't continue with stale broadcast addresses.
# shellcheck disable=SC2086
(cd bridge/evm && forge script script/DeployBridge.s.sol:DeployBridge \
    --rpc-url "$BASE_SEPOLIA_RPC" \
    --private-key "$DEPLOYER_PK" \
    --broadcast $verify_flag) 2>&1 | tee /tmp/deploy_base_sepolia.log | grep -E "^(BRIDGE_|SOMA_BRIDGE_|DEPLOYMENT_BLOCK|✓)"
# `set -eo pipefail` (set at the top of this script) ensures a non-zero
# exit from the forge subshell propagates out and aborts the script.

# ----- Parse broadcast for deployed addresses + emit env-var stanza -----
broadcast_json="bridge/evm/broadcast/DeployBridge.s.sol/${DEPLOY_EVM_CHAIN_ID}/run-latest.json"
if [ ! -f "$broadcast_json" ]; then
    echo ""
    echo "WARNING: broadcast file $broadcast_json not found. Re-run with"
    echo "         --resume if the deploy partially succeeded, or check the"
    echo "         RPC for the txs."
    exit 1
fi

echo ""
echo "============================================================"
echo "  Deployment artifacts (paste into validator configs):"
echo "============================================================"

# The broadcast file lists every tx in execution order. The proxies are
# the ERC1967Proxy contracts created by Upgrades.deployUUPSProxy.
# Pulling them by index matches the order in DeployBridge.s.sol::run().
python3 <<EOF
import json
with open("$broadcast_json") as f:
    d = json.load(f)

deployments = [
    t for t in d.get("transactions", [])
    if t.get("transactionType") == "CREATE" and t.get("contractAddress")
]

# Pattern matches DeployBridge.s.sol::run() order:
#   impl, proxy, vault, impl, proxy, impl, proxy
# So proxies are at indices 1, 3 (wait — vault is non-upgradeable, 1 deploy)
# Order is: commCommImpl, commProxy, vault, limImpl, limProxy, briImpl, briProxy
mapping = {
    "BRIDGE_COMMITTEE_IMPL":     0,
    "BRIDGE_COMMITTEE_PROXY":    1,
    "BRIDGE_VAULT":              2,
    "BRIDGE_LIMITER_IMPL":       3,
    "BRIDGE_LIMITER_PROXY":      4,
    "SOMA_BRIDGE_IMPL":          5,
    "SOMA_BRIDGE_PROXY":         6,
}

for label, idx in mapping.items():
    if idx < len(deployments):
        print(f"{label}={deployments[idx]['contractAddress']}")
    else:
        print(f"# {label}: MISSING (broadcast had only {len(deployments)} create txs)")
EOF

echo ""
echo "============================================================"
echo "  Next steps:"
echo "============================================================"
echo "  1. Source the addresses above into env vars."
echo "  2. Render per-validator configs from"
echo "     bridge-node/configs/base-sepolia.toml.template"
echo "  3. Fund each operator wallet from the CDP faucet."
echo "  4. Start each validator's bridge-node:"
echo "       cargo run --bin bridge-node -- --config /path/to/validator.toml"
echo "  5. Run inbound smoke test:"
echo "       # Get test USDC from Circle's faucet, then:"
echo "       cast send \$USDC 'approve(address,uint256)' \$SOMA_BRIDGE_PROXY 1000000 --rpc-url \$BASE_SEPOLIA_RPC --private-key \$USER_PK"
echo "       cast send \$SOMA_BRIDGE_PROXY 'deposit(uint8,bytes32,uint64)' 2 0x... 1000000 --rpc-url \$BASE_SEPOLIA_RPC --private-key \$USER_PK"
echo "       # Watch bridge-node logs for action observation + signing."
echo ""
echo "  Full per-step runbook: bridge/BASE_SEPOLIA_DEPLOY.md"
