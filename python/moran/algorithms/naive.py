"""Naive Monte Carlo -- Diaz et al. 2014 (Algorithmica, Theorem 13).

Simulates all birth-death steps including ineffective ones.
Absorption time bound: O(r/|r-1| * n^4).
"""

from .._moran import FixationResult, algorithms as _alg


def estimate(
        graph,
        r: float,
        *,
        epsilon: float = 0.1,
        delta: float = 0.25,
        seed: int = 0,
        num_threads: int = 0,
) -> FixationResult:
    return _alg.graph_structured.naive_mc_fixation(
        graph, r, epsilon=epsilon, delta=delta,
        seed=seed, num_threads=num_threads)
