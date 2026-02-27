"""Example: stake and unstake SOMA tokens with a validator on localnet.

Demonstrates:
  - get_latest_system_state: discover active validators
  - add_stake:               delegate 1 SOMA to a validator
  - list_owned_objects:      find StakedSoma objects
  - advance_epoch:           activate pending stake (admin/localnet only)
  - withdraw_stake:          unstake and reclaim tokens

Usage:
    1. Start a local network:   soma start localnet --force-regenesis
    2. Run:                     uv run soma-example-staking

Note: advance_epoch() is only available on localnet (requires admin_url).
On mainnet/testnet, stake activates at the next epoch boundary automatically.
"""

import asyncio

from soma_sdk import Keypair, SomaClient

async def run():
    client = await SomaClient(chain="localnet")

    kp = Keypair.generate()
    sender = kp.address()
    print(f"Staker address: {sender}")

    # -- Step 1: Fund account --------------------------------------------------
    print("\n=== Step 1: Fund account ===")
    await client.request_faucet(sender)
    balance = await client.get_balance(sender)
    print(f"Balance: {balance:.2f} SOMA")

    # -- Step 2: Discover validators -------------------------------------------
    print("\n=== Step 2: Discover validators ===")
    state = await client.get_latest_system_state()
    print(f"Epoch: {state.epoch}")
    print(f"Active validators: {len(state.validators.validators)}")
    for i, v in enumerate(state.validators.validators):
        pool = v.staking_pool
        pool_soma = SomaClient.to_soma(pool.soma_balance)
        print(
            f"  [{i}] {v.metadata.soma_address[:20]}..."
            f"  stake={pool_soma:.0f} SOMA"
            f"  commission={v.commission_rate / 100:.1f}%"
        )

    validator_addr = state.validators.validators[0].metadata.soma_address
    print(f"\nUsing validator: {validator_addr}")

    # -- Step 3: Stake 1 SOMA -------------------------------------------------
    print("\n=== Step 3: Stake 1 SOMA ===")
    balance_before = await client.get_balance(sender)

    await client.add_stake(
        signer=kp,
        validator=validator_addr,
        amount=1.0,
    )

    balance_after = await client.get_balance(sender)
    print(f"Balance: {balance_before:.2f} -> {balance_after:.2f} SOMA")

    # -- Step 4: Verify staked object ------------------------------------------
    print("\n=== Step 4: Verify staked object ===")
    staked_objects = await client.list_owned_objects(
        sender, object_type="staked_soma"
    )
    print(f"StakedSoma objects owned: {len(staked_objects)}")
    for s in staked_objects:
        print(f"  id={s.id}  version={s.version}")

    staked_id = staked_objects[0].id

    # -- Step 5: Advance epoch (stake activates) -------------------------------
    print("\n=== Step 5: Advance epoch (activates pending stake) ===")
    new_epoch = await client.advance_epoch()
    print(f"Epoch -> {new_epoch}")

    state = await client.get_latest_system_state()
    v = state.validators.validators[0]
    pool_soma = SomaClient.to_soma(v.staking_pool.soma_balance)
    print(f"Validator pool balance: {pool_soma:.0f} SOMA")

    # -- Step 6: Withdraw stake ------------------------------------------------
    print("\n=== Step 6: Withdraw stake ===")
    balance_before = await client.get_balance(sender)

    await client.withdraw_stake(signer=kp, staked_soma_id=staked_id)

    balance_after = await client.get_balance(sender)
    print(f"Balance: {balance_before:.2f} -> {balance_after:.2f} SOMA")

    staked_objects = await client.list_owned_objects(
        sender, object_type="staked_soma"
    )
    print(f"StakedSoma objects remaining: {len(staked_objects)}")

    print("\n=== Staking lifecycle complete! ===")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
