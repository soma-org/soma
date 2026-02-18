"""Example: request test tokens from a local faucet and check balance.

Usage:
    1. Start a local network with faucet:   soma start --force-regenesis --with-faucet
    2. Run this example:                     uv run soma-example-faucet <address>

If no address is supplied, the active address from ~/.soma/client.yaml is used.
"""

import asyncio
import json
import os
import sys

from soma_sdk import SomaClient, WalletContext, request_faucet

RPC_URL = os.environ.get("SOMA_RPC_URL", "http://localhost:9000")
WALLET_CONFIG = os.environ.get(
    "SOMA_WALLET_CONFIG",
    os.path.expanduser("~/.soma/client.yaml"),
)


async def run():
    # Resolve address: CLI arg > wallet active address.
    if len(sys.argv) > 1:
        address = sys.argv[1]
    else:
        wallet = WalletContext(WALLET_CONFIG)
        address = await wallet.active_address()
    print(f"Recipient address: {address}")

    # Request tokens from the local faucet.
    print("Requesting tokens from faucet...")
    resp = json.loads(await request_faucet(address))
    status = resp.get("status")
    if status == "Success":
        coins = resp.get("coins_sent", [])
        print(f"Received {len(coins)} coin(s):")
        for coin in coins:
            print(f"  - {coin['amount']} SOMA (id: {coin['id']})")
    else:
        print(f"Faucet request failed: {status}")
        return

    # Check balance via RPC.
    client = await SomaClient(RPC_URL)
    balance = json.loads(await client.get_balance(address))
    total = balance.get("totalBalance", balance.get("total_balance", "0"))
    print(f"Balance after faucet: {total} SOMA")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
