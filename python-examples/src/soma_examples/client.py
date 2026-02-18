"""Example: connect to a Soma node and query basic chain state."""

import asyncio
import json

from soma_sdk import SomaClient


async def run():
    client = await SomaClient("http://localhost:9000")

    chain_id = await client.get_chain_identifier()
    version = await client.get_server_version()
    protocol = await client.get_protocol_version()
    print(f"Chain: {chain_id}")
    print(f"Server version: {version}")
    print(f"Protocol version: {protocol}")

    state = json.loads(await client.get_latest_system_state())
    print(f"Epoch: {state.get('epoch')}")

    cp = json.loads(await client.get_latest_checkpoint())
    print(f"Latest checkpoint: {cp.get('sequence_number')}")

    targets = json.loads(await client.list_targets(status="open", limit=5))
    print(f"Open targets: {len(targets)}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
