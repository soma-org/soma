"""Example: request test tokens from a local faucet and check balance.

Usage:
    1. Start a local network:   soma start --force-regenesis
    2. Run this example:         uv run soma-example-faucet [address]

If no address is supplied, the active address from ~/.soma/soma_config/client.yaml is used.
"""

import asyncio
import os
import sys

from soma_sdk import SomaClient, Wallet, request_faucet, to_soma

RPC_URL = os.environ.get("SOMA_RPC_URL", "http://localhost:9000")
WALLET_CONFIG = os.environ.get(
    "SOMA_WALLET_CONFIG",
    "~/.soma/soma_config/client.yaml",
)


async def run():
    client = await SomaClient(RPC_URL)
    wallet = Wallet(WALLET_CONFIG, client)

    # Resolve address: CLI arg > wallet active address.
    if len(sys.argv) > 1:
        address = sys.argv[1]
    else:
        address = await wallet.active_address()
    print(f"Recipient address: {address}")

    # Request tokens from the local faucet.
    print("Requesting tokens from faucet...")
    resp = request_faucet(address)
    status = resp.get("status")
    if status != "Success":
        print(f"Faucet request failed: {status}")
        return

    coins = resp.get("coins_sent", [])
    total_received = sum(c["amount"] for c in coins)
    print(f"Received {to_soma(total_received):.0f} SOMA across {len(coins)} coin(s)")

    # Check balance via RPC.
    balance = await client.get_balance(address)
    print(f"Total balance: {to_soma(balance):.0f} SOMA")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
