"""Example: request test tokens from a local faucet and check balance.

Usage:
    1. Start a local network with faucet:   soma start --force-regenesis --with-faucet
    2. Run this example:                     uv run soma-example-faucet <address>

If no address is supplied, the active address from ~/.soma/soma_config/client.yaml is used.
"""

import asyncio
import json
import os
import sys
import urllib.request

from soma_sdk import SomaClient, WalletContext

RPC_URL = os.environ.get("SOMA_RPC_URL", "http://localhost:9000")
FAUCET_URL = os.environ.get("SOMA_FAUCET_URL", "http://127.0.0.1:9123/gas")
SHANNONS_PER_SOMA = 1_000_000_000
WALLET_CONFIG = os.environ.get(
    "SOMA_WALLET_CONFIG",
    os.path.expanduser("~/.soma/soma_config/client.yaml"),
)


def request_faucet(address: str, url: str = FAUCET_URL) -> dict:
    """Request tokens from a faucet server (simple HTTP POST)."""
    body = json.dumps({"FixedAmountRequest": {"recipient": address}}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


async def run():
    # Resolve address: CLI arg > wallet active address.
    if len(sys.argv) > 1:
        address = sys.argv[1]
    else:
        wallet = WalletContext(WALLET_CONFIG)
        address = await wallet.active_address()
    print(f"Recipient address: {address}")

    # Request tokens from the local faucet.
    print(f"Requesting tokens from faucet at {FAUCET_URL}...")
    resp = request_faucet(address)
    status = resp.get("status")
    if status != "Success":
        print(f"Faucet request failed: {status}")
        return

    coins = resp.get("coins_sent", [])
    total_received = sum(c["amount"] for c in coins)
    print(f"Received {total_received / SHANNONS_PER_SOMA:.0f} SOMA across {len(coins)} coin(s)")

    # Check balance via RPC.
    client = await SomaClient(RPC_URL)
    balance = await client.get_balance(address)
    print(f"Total balance: {balance / SHANNONS_PER_SOMA:.0f} SOMA")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
