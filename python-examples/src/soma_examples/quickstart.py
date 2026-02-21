"""Quickstart: Complete Soma mining lifecycle on localnet.

Start a local network (one command):

    soma start --force-regenesis --small-model

Then run this example:

    uv run soma-example-quickstart
"""

import asyncio
import os

from safetensors.numpy import save as save_safetensors

from soma_sdk import (
    ScoringClient,
    SomaClient,
    Wallet,
    advance_epoch,
    encrypt_weights,
    request_faucet,
)
from soma_examples.model_utils import build_model_weights
from soma_examples.local_storage import LocalStorage


async def run():
    # Connect to localnet
    client = await SomaClient("http://127.0.0.1:9000")
    wallet = Wallet("~/.soma/soma_config/client.yaml", client)
    sender = await wallet.active_address()
    print(f"Sender: {sender}")

    # -- Step 1: Fund account --------------------------------------------------
    print("\n=== Step 1: Fund account ===")
    request_faucet(sender)
    await asyncio.sleep(2)  # wait for faucet tx
    print(f"Balance: {await wallet.get_balance():.2f} SOMA")

    # -- Step 2: Upload model weights + test data ------------------------------
    print("\n=== Step 2: Upload model & data ===")
    with LocalStorage() as storage:
        model_bytes = save_safetensors(build_model_weights(seed=42))
        encrypted_bytes, decryption_key = encrypt_weights(model_bytes)
        model_url = storage.upload("weights.safetensors.enc", encrypted_bytes)
        print(f"Model uploaded ({len(encrypted_bytes)} bytes, encrypted)")

        test_data = os.urandom(1024)
        data_url = storage.upload("data.bin", test_data)
        print("Test data uploaded (1024 bytes)")

        # -- Step 3: Commit model on-chain -------------------------------------
        print("\n=== Step 3: Commit model ===")
        model_id = await wallet.commit_model(
            weights_url=model_url,
            encrypted_weights=encrypted_bytes,
            commission_rate=1000,  # 10%
        )
        print(f"Model committed: {model_id}")

        # -- Step 4: Advance epoch, then reveal --------------------------------
        print("\n=== Step 4: Advance epoch + reveal ===")
        print(f"Epoch -> {advance_epoch()}")

        await wallet.reveal_model(
            model_id=model_id,
            weights_url=model_url,
            encrypted_weights=encrypted_bytes,
            decryption_key=decryption_key,
            embedding=[0.1] * await wallet.embedding_dim(),
        )
        print("Model revealed")

        # -- Step 5: Advance epoch (targets created at boundary) ---------------
        print("\n=== Step 5: Advance epoch (targets generated) ===")
        print(f"Epoch -> {advance_epoch()}")

        # -- Step 6: Find target, fetch models from chain, score, submit -------
        print("\n=== Step 6: Score + submit ===")
        targets = await wallet.get_targets(status="open")
        target = next(t for t in targets if model_id in t.model_ids)
        print(f"Target: {target.id}  threshold={target.distance_threshold}")

        # Fetch model manifests from on-chain state (as a real miner would)
        models = await wallet.get_model_manifests(target)
        print(f"Fetched {len(models)} model manifest(s) from chain")

        print("Scoring (this may take a minute on CPU)...", flush=True)
        score = ScoringClient().score(
            data_url=data_url,
            data=test_data,
            models=models,
            target_embedding=target.embedding,
        )
        winner = models[score.winner]
        print(f"Winner: {winner.url}  distance={score.distance[score.winner]:.6f}")

        await wallet.submit_data(
            target_id=target.id,
            data=test_data,
            data_url=data_url,
            model_id=target.model_ids[score.winner],
            embedding=score.embedding,
            distance_score=score.distance[score.winner],
        )
        print("Data submitted")

        # -- Step 7: Skip challenge window (2 epoch advances) ------------------
        print("\n=== Step 7: Skip challenge window ===")
        print(f"Epoch -> {advance_epoch()}")
        print(f"Epoch -> {advance_epoch()}")

        # -- Step 8: Claim reward ----------------------------------------------
        print("\n=== Step 8: Claim reward ===")
        print(f"Target reward pool: {target.reward_pool / 1_000_000_000:.4f} SOMA")
        balance_before = await wallet.get_balance()
        await wallet.claim_rewards(target_id=target.id)
        print(f"Claimed: {await wallet.get_balance() - balance_before:.4f} SOMA")
        print("\n=== Quickstart complete! ===")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
