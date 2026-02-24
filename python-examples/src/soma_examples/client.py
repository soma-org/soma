"""Example: connect to a SOMA node and query basic chain state."""

import asyncio

from soma_sdk import SomaClient


async def run():
    client = await SomaClient("http://localhost:9000")

    chain_id = await client.get_chain_identifier()
    version = await client.get_server_version()
    protocol = await client.get_protocol_version()
    print(f"Chain: {chain_id}")
    print(f"Server version: {version}")
    print(f"Protocol version: {protocol}")

    state = await client.get_latest_system_state()
    print(f"Epoch: {state.epoch}")

    cp = await client.get_latest_checkpoint()
    print(f"Latest checkpoint: {cp.data.sequence_number}")

    targets = await client.get_targets(status="open", limit=5)
    print(f"Open targets: {len(targets)}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
