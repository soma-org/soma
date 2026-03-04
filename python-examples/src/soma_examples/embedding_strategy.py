"""Embedding Strategy: Find the optimal model embedding to maximize target selection.

Simulates the on-chain stake-weighted KNN model selection algorithm across
many random targets to find the embedding position that would give a new
competing model the highest probability of being selected.

On-chain algorithm (types/src/model_selection.rs):
  - Target embeddings are drawn from N(0, 1) per dimension
  - For each target, models are scored: weighted_score = dist^2 / voting_power
  - voting_power = model_stake / total_stake (normalized to sum to 1.0)
  - Top-k models with lowest weighted_score are selected (k = models_per_target)

Start a local network:

    soma start --force-regenesis --small-model

Then run this example:

    uv run soma-example-embedding-strategy

Or against testnet:

    uv run soma-example-embedding-strategy --chain testnet
"""

import argparse
import asyncio

import numpy as np

from soma_sdk import SomaClient


def simulate_selection_rate(
    candidate_embeddings: np.ndarray,
    existing_embeddings: np.ndarray,
    existing_stakes: np.ndarray,
    candidate_stake: float,
    models_per_target: int,
    n_targets: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate selection rate for each candidate embedding via Monte Carlo.

    Generates random target embeddings from N(0, 1) (matching on-chain
    target generation) and computes how often each candidate would be
    selected using the stake-weighted KNN algorithm.

    Args:
        candidate_embeddings: (n_candidates, dim) array of candidate positions
        existing_embeddings: (n_existing, dim) array of current model embeddings
        existing_stakes: (n_existing,) array of stakes in SOMA
        candidate_stake: stake for the hypothetical new model in SOMA
        models_per_target: k value for top-k selection
        n_targets: number of random targets to simulate
        seed: random seed

    Returns:
        (n_candidates,) array of selection rates in [0, 1]
    """
    rng = np.random.default_rng(seed)
    dim = existing_embeddings.shape[1]
    n_existing = existing_embeddings.shape[0]
    n_candidates = candidate_embeddings.shape[0]

    # Pre-compute existing voting powers (without candidate)
    all_stakes = np.append(existing_stakes, candidate_stake)
    total_stake = all_stakes.sum()
    voting_powers = all_stakes / total_stake  # (n_existing + 1,)

    # Generate random target embeddings from N(0, 1) — matches on-chain generation
    targets = rng.standard_normal((n_targets, dim)).astype(np.float32)

    # Compute squared distances from targets to existing models: (n_targets, n_existing)
    # ||t - e||^2 = ||t||^2 + ||e||^2 - 2 * t @ e^T
    existing_sq = np.sum(existing_embeddings**2, axis=1)  # (n_existing,)
    targets_sq = np.sum(targets**2, axis=1, keepdims=True)  # (n_targets, 1)
    existing_dists_sq = (
        targets_sq + existing_sq[np.newaxis, :] - 2.0 * targets @ existing_embeddings.T
    )  # (n_targets, n_existing)

    # Weighted scores for existing models: (n_targets, n_existing)
    existing_scores = existing_dists_sq / (voting_powers[:n_existing] + 1e-10)

    # For each candidate, compute its weighted score and count selections
    candidate_vp = voting_powers[-1]  # candidate's voting power
    selection_counts = np.zeros(n_candidates, dtype=np.int64)

    # Process candidates in batches for memory efficiency
    batch_size = 100
    for start in range(0, n_candidates, batch_size):
        end = min(start + batch_size, n_candidates)
        batch = candidate_embeddings[start:end]  # (batch, dim)

        # Squared distances from targets to candidate batch: (n_targets, batch)
        batch_sq = np.sum(batch**2, axis=1)  # (batch,)
        cand_dists_sq = targets_sq + batch_sq[np.newaxis, :] - 2.0 * targets @ batch.T

        # Weighted score for each candidate: (n_targets, batch)
        cand_scores = cand_dists_sq / (candidate_vp + 1e-10)

        # For each target, check if candidate is in top-k across all models
        for j in range(end - start):
            # Combine existing scores with this candidate's scores
            all_scores = np.column_stack(
                [existing_scores, cand_scores[:, j : j + 1]]
            )  # (n_targets, n_existing + 1)

            # Candidate is the last column. Check if it's in top-k.
            # argsort ascending, take first k indices, check if candidate index is present
            candidate_idx = n_existing
            topk_indices = np.argpartition(all_scores, models_per_target, axis=1)[
                :, :models_per_target
            ]
            selected = np.any(topk_indices == candidate_idx, axis=1)
            selection_counts[start + j] = selected.sum()

    return selection_counts / n_targets


async def run(chain: str = "localnet"):
    client = await SomaClient(chain=chain)

    # Fetch active model embeddings
    print("=== Fetching active model embeddings ===")
    models = await client.get_active_models()

    if not models:
        print("No active models found. Register a model first (see quickstart.py).")
        return

    print(f"Found {len(models)} active model(s):\n")
    for m in models:
        print(
            f"  Model {m.model_id[:16]}...  stake={m.stake:.2f} SOMA  "
            f"embedding_norm={np.linalg.norm(m.embedding):.4f}"
        )

    # Get system parameters
    state = await client.get_latest_system_state()
    models_per_target = state.parameters.target_models_per_target
    dim = state.parameters.target_embedding_dim
    print(f"\nEmbedding dim: {dim}")
    print(f"Models per target (k): {models_per_target}")

    # Prepare existing model data as numpy arrays
    existing_embeddings = np.array([m.embedding for m in models], dtype=np.float32)
    existing_stakes = np.array([m.stake for m in models], dtype=np.float64)

    # Hypothetical new model stake (minimum: 1 SOMA)
    candidate_stake = 1.0
    print(f"\nSimulating with new model stake: {candidate_stake:.2f} SOMA")

    # Generate candidate embeddings at the same scale as existing models.
    # Raw N(0,1) vectors have norm ≈ sqrt(dim), but trained model embeddings
    # live at a different scale. We sample random directions and rescale to
    # match the mean norm of existing models so candidates can compete fairly.
    n_candidates = 500
    n_targets = 5000
    existing_norms = np.linalg.norm(existing_embeddings, axis=1)
    mean_norm = existing_norms.mean()
    print(f"\nExisting model mean norm: {mean_norm:.4f}")
    print(
        f"Generating {n_candidates} candidate positions, simulating {n_targets} targets..."
    )

    rng = np.random.default_rng(0)
    candidates = rng.standard_normal((n_candidates, dim)).astype(np.float32)
    candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    candidates = candidates * (mean_norm / candidate_norms)

    # Simulate selection rate for each candidate
    rates = simulate_selection_rate(
        candidate_embeddings=candidates,
        existing_embeddings=existing_embeddings,
        existing_stakes=existing_stakes,
        candidate_stake=candidate_stake,
        models_per_target=models_per_target,
        n_targets=n_targets,
    )

    # Report results
    best_idx = np.argmax(rates)
    best_rate = rates[best_idx]
    best_embedding = candidates[best_idx]

    print("\n=== Results ===")
    print(f"Best candidate selection rate: {best_rate:.1%}")
    print(f"Mean candidate selection rate: {rates.mean():.1%}")
    print(f"Worst candidate selection rate: {rates.min():.1%}")

    # Baseline: what rate would a uniform-stake model get if placed randomly?
    # With N models + 1 candidate all equal stake, expected rate = k / (N+1)
    n_models = len(models)
    uniform_baseline = models_per_target / (n_models + 1)
    print(f"\nUniform baseline (equal stake, random position): {uniform_baseline:.1%}")
    print(
        f"Best candidate advantage over baseline: {best_rate / uniform_baseline:.2f}x"
    )

    # Show distance from best candidate to each existing model
    print("\nBest candidate distances to existing models:")
    for m in models:
        dist = np.sqrt(np.sum((best_embedding - np.array(m.embedding)) ** 2))
        print(f"  -> {m.model_id[:16]}...  euclidean_dist={dist:.4f}")

    print(f"\nOptimal embedding (first 8 dims): {best_embedding[:8].tolist()}")
    print(f"Full embedding has {dim} dimensions.")


def main():
    parser = argparse.ArgumentParser(description="Find optimal model embedding")
    parser.add_argument(
        "--chain", type=str, default="localnet", help="Chain to connect to"
    )
    args = parser.parse_args()
    asyncio.run(run(chain=args.chain))


if __name__ == "__main__":
    main()
