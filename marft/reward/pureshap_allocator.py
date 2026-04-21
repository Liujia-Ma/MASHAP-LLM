import itertools
import math
from typing import Callable

import numpy as np


Coalition = tuple[int, ...]


def _safe_score(score) -> float:
    if isinstance(score, (int, float, bool, np.number)):
        return float(score)
    if isinstance(score, (list, tuple)) and len(score) > 0:
        return float(score[0])
    return float(score)


def pureshap_generate_all_coalitions(num_agents: int) -> list[Coalition]:
    """Enumerate all 2^N coalitions as sorted index tuples."""
    return [
        tuple(i for i in range(num_agents) if (mask >> i) & 1)
        for mask in range(1 << num_agents)
    ]


def pureshap_gather_coalition_scores(
    *,
    num_agents: int,
    coalition_score_fn: Callable[[Coalition], float],
    coalition_has_actor_fn: Callable[[Coalition], bool],
    full_coalition_score: float,
) -> tuple[dict[Coalition, float], dict]:
    """
    Build the exact coalition value table with pruning:
    1) Full coalition uses cache score (no re-rollout).
    2) Coalitions without actor are hardcoded to 0.
    3) Coalitions with actor are physically rolled out + evaluated.
    """
    full_coalition = tuple(range(num_agents))
    coalition_scores: dict[Coalition, float] = {full_coalition: float(full_coalition_score)}
    rollout_evaluated = 0
    hardcoded_zero = 0

    for coalition in pureshap_generate_all_coalitions(num_agents):
        if coalition in coalition_scores:
            continue
        if len(coalition) == 0 or not coalition_has_actor_fn(coalition):
            coalition_scores[coalition] = 0.0
            hardcoded_zero += 1
            continue
        coalition_scores[coalition] = _safe_score(coalition_score_fn(coalition))
        rollout_evaluated += 1

    expected_total = 1 << num_agents
    if len(coalition_scores) != expected_total:
        raise RuntimeError(
            f"Incomplete coalition table: got {len(coalition_scores)}, expected {expected_total}."
        )
    stats = {
        "coalitions_total": int(expected_total),
        "coalitions_rollout_evaluated": int(rollout_evaluated),
        "coalitions_hardcoded_zero": int(hardcoded_zero),
        "full_coalition_cache_hit": 1,
    }
    return coalition_scores, stats


def pureshap_compute_exact_shapley_values(
    *,
    num_agents: int,
    coalition_scores: dict[Coalition, float],
) -> np.ndarray:
    """Compute exact Shapley values from a complete coalition score table."""
    factorial_n = math.factorial(num_agents)
    phi = np.zeros(num_agents, dtype=np.float64)

    def value_of(coalition: Coalition) -> float:
        return float(coalition_scores[tuple(sorted(coalition))])

    for agent_idx in range(num_agents):
        others = [a for a in range(num_agents) if a != agent_idx]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                subset = tuple(sorted(subset))
                subset_with_agent = tuple(sorted(subset + (agent_idx,)))
                weight = (
                    math.factorial(len(subset))
                    * math.factorial(num_agents - len(subset) - 1)
                    / factorial_n
                )
                marginal = value_of(subset_with_agent) - value_of(subset)
                phi[agent_idx] += weight * marginal
    return phi


class PureShapleyAllocator:
    def __init__(self, *, efficiency_tol: float = 1e-5):
        self.efficiency_tol = efficiency_tol

    def allocate(
        self,
        *,
        num_agents: int,
        coalition_score_fn: Callable[[Coalition], float],
        coalition_has_actor_fn: Callable[[Coalition], bool],
        full_coalition_score: float,
    ) -> tuple[np.ndarray, dict]:
        coalition_scores, gather_stats = pureshap_gather_coalition_scores(
            num_agents=num_agents,
            coalition_score_fn=coalition_score_fn,
            coalition_has_actor_fn=coalition_has_actor_fn,
            full_coalition_score=float(full_coalition_score),
        )

        phi = pureshap_compute_exact_shapley_values(
            num_agents=num_agents,
            coalition_scores=coalition_scores,
        )
        phi = phi.astype(np.float32)

        shapley_sum = float(np.sum(phi))
        efficiency_gap = shapley_sum - float(full_coalition_score)
        if not math.isclose(
            shapley_sum,
            float(full_coalition_score),
            rel_tol=1e-6,
            abs_tol=self.efficiency_tol,
        ):
            raise AssertionError(
                "PureShap efficiency check failed: "
                f"sum(phi)={shapley_sum:.8f}, full_score={float(full_coalition_score):.8f}"
            )

        debug = {
            "allocation_mode": "pureshap",
            "shapley_method": "exact_pureshap",
            "total_score": float(full_coalition_score),
            "shapley_sum": shapley_sum,
            "efficiency_gap": float(efficiency_gap),
            **gather_stats,
        }
        return phi, debug
