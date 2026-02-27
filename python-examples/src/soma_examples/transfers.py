"""Example: transfer SOMA coins between accounts on localnet.

Demonstrates:
  - transfer_coin:      send SOMA from one account to another
  - pay_coins:          send to multiple recipients in a single transaction
  - get_balance:        check account balances before and after
  - list_owned_objects: inspect owned coin objects

Usage:
    1. Start a local network:   soma start localnet --force-regenesis
    2. Run:                     uv run soma-example-transfers
"""

import asyncio

from soma_sdk import Keypair, SomaClient

async def run():
    client = await SomaClient(chain="localnet")

    sender_kp = Keypair.generate()
    sender = sender_kp.address()
    recipient1 = Keypair.generate().address()
    recipient2 = Keypair.generate().address()
    print(f"Sender:     {sender}")
    print(f"Recipient1: {recipient1}")
    print(f"Recipient2: {recipient2}")

    # -- Step 1: Fund sender via faucet ----------------------------------------
    print("\n=== Step 1: Fund sender account ===")
    await client.request_faucet(sender)
    balance = await client.get_balance(sender)
    print(f"Sender balance: {balance:.2f} SOMA")

    # -- Step 2: Transfer coins to recipient1 ----------------------------------
    print("\n=== Step 2: Transfer 0.5 SOMA to recipient1 ===")
    balance_before = await client.get_balance(sender)

    await client.transfer_coin(
        signer=sender_kp,
        recipient=recipient1,
        amount=0.5,
    )

    sender_after = await client.get_balance(sender)
    recip1_bal = await client.get_balance(recipient1)
    print(
        f"Sender balance:     {balance_before:.2f}"
        f" -> {sender_after:.2f} SOMA"
    )
    print(f"Recipient1 balance: {recip1_bal:.2f} SOMA")

    # -- Step 3: Multi-send with pay_coins -------------------------------------
    print("\n=== Step 3: Pay 0.25 SOMA to each of two recipients ===")
    balance_before = await client.get_balance(sender)

    await client.pay_coins(
        signer=sender_kp,
        recipients=[recipient1, recipient2],
        amounts=[0.25, 0.25],
    )

    sender_after = await client.get_balance(sender)
    recip1_bal = await client.get_balance(recipient1)
    recip2_bal = await client.get_balance(recipient2)
    print(
        f"Sender balance:     {balance_before:.2f}"
        f" -> {sender_after:.2f} SOMA"
    )
    print(f"Recipient1 balance: {recip1_bal:.2f} SOMA (cumulative)")
    print(f"Recipient2 balance: {recip2_bal:.2f} SOMA")

    # -- Step 4: Inspect owned objects -----------------------------------------
    print("\n=== Step 4: List sender's coin objects ===")
    coins = await client.list_owned_objects(sender, object_type="coin")
    print(f"Sender owns {len(coins)} coin object(s):")
    for c in coins:
        print(f"  {c.id}  (v{c.version})")

    print("\n=== Transfers complete! ===")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
