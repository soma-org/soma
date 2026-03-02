"""Quickstart: Complete SOMA data submission lifecycle on localnet.

Start a local network (one command):

    soma start --force-regenesis --small-model

Then run this example:

    uv run soma-example-quickstart
"""

import asyncio
import os

from safetensors.numpy import save as save_safetensors

from soma_sdk import Keypair, SomaClient
from soma_examples.model_utils import build_model_weights
from soma_examples.local_storage import LocalStorage


async def run():
    # Connect to localnet (with scoring, admin, faucet gRPC services)
    kp = Keypair.generate()
    client = await SomaClient(chain="localnet")
    sender = kp.address()
    print(f"Sender: {sender}")

    # -- Step 1: Fund account --------------------------------------------------
    print("\n=== Step 1: Fund account ===")
    await client.request_faucet(sender)
    balance = await client.get_balance(sender)
    print(f"Balance: {balance:.2f} SOMA")

    # -- Step 2: Upload model weights + test data ------------------------------
    print("\n=== Step 2: Upload model & data ===")
    with LocalStorage() as storage:
        model_bytes = save_safetensors(build_model_weights(seed=42))
        encrypted_bytes, decryption_key = SomaClient.encrypt_weights(model_bytes)
        model_url = storage.upload("weights.safetensors.enc", encrypted_bytes)
        print(f"Model uploaded ({len(encrypted_bytes)} bytes, encrypted)")

        test_data = os.urandom(1024)
        data_url = storage.upload("data.bin", test_data)
        print("Test data uploaded (1024 bytes)")

        # -- Step 3: Commit model on-chain -------------------------------------
        print("\n=== Step 3: Commit model ===")
        embedding_dim = await client.get_embedding_dim()
        embedding = [0.1] * embedding_dim
        await client.commit_model(
            signer=kp,
            weights_url=model_url,
            encrypted_weights=encrypted_bytes,
            decryption_key=decryption_key,
            embedding=embedding,
            commission_rate=1000,  # 10%
        )
        print("Model committed")

        # -- Step 4: Advance epoch, then reveal --------------------------------
        print("\n=== Step 4: Advance epoch + reveal ===")
        epoch = await client.advance_epoch()
        print(f"Epoch -> {epoch}")

        # Discover the model_id from the pending models in system state
        state = await client.get_latest_system_state()
        pending = vars(state.model_registry.pending_models)
        model_id = next(
            mid for mid, m in pending.items()
            if m.owner == sender
        )
        await client.reveal_model(
            signer=kp,
            model_id=model_id,
            decryption_key=decryption_key,
            embedding=embedding,
        )
        print(f"Model revealed: {model_id}")

        # -- Step 5: Advance epoch (targets created at boundary) ---------------
        print("\n=== Step 5: Advance epoch (targets generated) ===")
        epoch = await client.advance_epoch()
        print(f"Epoch -> {epoch}")

        # -- Step 6: Find target, fetch models from chain, score, submit -------
        print("\n=== Step 6: Score + submit ===")
        targets = await client.get_targets(status="open")
        target = next(t for t in targets if model_id in t.model_ids)
        print(f"Target: {target.id}  threshold={target.distance_threshold}")

        # Fetch model manifests from on-chain state (as a real submitter would)
        manifests = await client.get_model_manifests(target)
        print(f"Fetched {len(manifests)} model manifest(s) from chain")

        print("Scoring (this may take a minute on CPU)...", flush=True)
        score = await client.score(
            data_url=data_url,
            models=manifests,
            target_embedding=target.embedding,
            data=test_data,
            seed=0,
        )
        print(
            f"Winner: index={score.winner}  distance={score.distance[score.winner]:.6f}"
        )

        await client.submit_data(
            signer=kp,
            target_id=target.id,
            data=test_data,
            data_url=data_url,
            model_id=target.model_ids[score.winner],
            embedding=score.embedding,
            distance_score=score.distance[score.winner],
            loss_score=score.loss_score,
        )
        print("Data submitted")

        # -- Step 7: Skip challenge window (2 epoch advances) ------------------
        print("\n=== Step 7: Skip challenge window ===")
        print(f"Epoch -> {await client.advance_epoch()}")
        print(f"Epoch -> {await client.advance_epoch()}")

        # -- Step 8: Claim reward ----------------------------------------------
        print("\n=== Step 8: Claim reward ===")
        print(f"Target reward pool: {SomaClient.to_soma(target.reward_pool):.4f} SOMA")
        balance_before = await client.get_balance(sender)
        await client.claim_rewards(signer=kp, target_id=target.id)
        balance_after = await client.get_balance(sender)
        print(f"Claimed: {balance_after - balance_before:.4f} SOMA")
        print("\n=== Quickstart complete! ===")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
