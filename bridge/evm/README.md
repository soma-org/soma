# bridge/evm

Ethereum-side contracts for the Soma ↔ Eth USDC bridge.

In-tree alongside [`bridge-node/`](../../bridge-node/) — mirroring [`MystenLabs/sui/bridge/evm/`](https://github.com/MystenLabs/sui/tree/main/bridge/evm)'s monorepo layout. A wire-format change in [`types/src/bridge.rs`](../../types/src/bridge.rs) and the matching Solidity update land in the same commit.

Same UUPS-upgradeable, ecrecover-based stake-weighted committee, same `BridgeMessage` library, same per-action stake thresholds, same V2 token-transfer payload (72 bytes with `timestampMs`), same 48h `isMatureMessage` limiter bypass. Adapted for Soma:

- **USDC-only** — no token registry, no multi-token branching, no WETH9
- **Multi-Soma-chain destination set** — `isChainSupported` mapping accepts mainnet/testnet/custom (Sui parity)
- **V2 token transfer payload** — 72-byte `(senderLen, sender, targetChain, targetLen, target, tokenType, amount, timestampMs)` matching Soma's `encode_withdraw_payload`

The Solidity wire format MUST stay in sync with [`types/src/bridge.rs::encode_bridge_message`](../../types/src/bridge.rs) — the off-chain bridge node signs the bytes this library reconstructs.

## Contracts

| File | Purpose |
|---|---|
| `BridgeMessage.sol` | Canonical message format + per-type stake thresholds + payload decoders |
| `BridgeCommittee.sol` | ecrecover-based committee with stake-weighted threshold + blocklist |
| `BridgeVault.sol` | USDC custody (owned by `SomaBridge`) |
| `BridgeLimiter.sol` | Sliding-24h-window USD rate limit on outbound transfers |
| `SomaBridge.sol` | User `deposit()` + quorum-signed `transferBridgedTokensWithSignatures` |
| `utils/MessageVerifier.sol` | Modifier: sig verify + chain-id + per-type nonce check |
| `utils/CommitteeUpgradeable.sol` | UUPS upgrade gated by quorum-signed `UPGRADE` message |

## Quick start

```bash
forge install   # OpenZeppelin contracts-upgradeable + foundry-upgrades
forge build
forge test      # 11 tests covering wire format + ecrecover + end-to-end deposit/withdraw
```

## Wire format

```
PREFIX || type(1) || version(1) || nonce(8 BE) || chainID(1) || payload
```

- `PREFIX` = `"SOMA_BRIDGE_MESSAGE"` (19 bytes)
- Message types: `USDC_DEPOSIT=0`, `USDC_WITHDRAW=1`, `EMERGENCY_OP=2`, `COMMITTEE_UPDATE=3`, `BLOCKLIST=4`, `UPDATE_LIMIT=5`, `UPGRADE=6`
- Chain ids: `SomaMainnet=0`, `SomaTestnet=1`, `SomaCustom=2`, `EthMainnet=10`, `EthSepolia=11`, `EthCustom=12`

## Stake thresholds (BPS, total = 10000)

| Action | BPS | Note |
|---|---|---|
| Token transfer | 3334 | ~33% |
| Freeze (emergency pause) | 450 | Intentionally cheap so a single watchdog can fire |
| Unfreeze | 5001 | Majority required to undo |
| Blocklist | 5001 | |
| Limit update | 5001 | |
| Upgrade | 5001 | |
| Committee update | 5001 | (on-chain handler deferred — see `BridgeMessage.sol`) |

## Deployment

High-level order (Foundry script in `script/`, TODO):

1. Deploy `BridgeCommittee` (initialize with `(members, stake, chainID)`).
2. Deploy `BridgeVault(USDC)`.
3. Deploy `BridgeLimiter` (initialize with `(committee, totalLimit)`).
4. Deploy `SomaBridge` (initialize with `(committee, USDC, vault, limiter, supportedChainIDs[])`). Typical values for `supportedChainIDs`:
   - Eth mainnet contract → `[SomaMainnet]`
   - Eth Sepolia contract → `[SomaTestnet, SomaCustom]`
5. `vault.transferOwnership(bridge)`.
6. `limiter.transferOwnership(bridge)`.

All UUPS-upgradeable; the proxy is replaced via a quorum-signed `UPGRADE` message through `CommitteeUpgradeable.upgradeWithSignatures`.

## Related

- Off-chain bridge node: [`../../bridge-node/`](../../bridge-node/)
- Wire format (Rust): [`../../types/src/bridge.rs`](../../types/src/bridge.rs)
- Reference architecture: [`MystenLabs/sui/bridge/evm/`](https://github.com/MystenLabs/sui/tree/main/bridge/evm) (V2, authoritative)
