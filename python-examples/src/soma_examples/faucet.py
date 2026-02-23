"""Example: request test tokens from a local faucet and check balance.

Usage:
    1. Start a local network:   soma start --force-regenesis
    2. Run:                     uv run soma-example-faucet [address]

If no address is supplied, a new keypair is generated.
"""

import asyncio
import sys

from soma_sdk import Keypair, SomaClient


async def run():
    client = await SomaClient(
        "http://localhost:9000",
        faucet_url="http://127.0.0.1:9123",
    )

    # Resolve address: CLI arg > generate new keypair.
    if len(sys.argv) > 1:
        address = sys.argv[1]
    else:
        kp = Keypair.generate()
        address = kp.address()
    print(f"Recipient address: {address}")

    # Request tokens from the local faucet.
    print("Requesting tokens from faucet...")
    resp = await client.request_faucet(address)
    if resp.status != "Success":
        print(f"Faucet request failed: {resp.status}")
        return

    total_received = sum(c.amount for c in resp.coins_sent)
    print(f"Received {SomaClient.to_soma(total_received):.0f} SOMA across {len(resp.coins_sent)} coin(s)")

    # Check balance via RPC.
    balance = await client.get_balance(address)
    print(f"Total balance: {SomaClient.to_soma(balance):.0f} SOMA")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
