"""Moran process fixation probability library."""

__version__ = "0.1.0"

from . import algorithms
from . import exact
from . import graph
from ._moran import (
    FixationResult,
    DegreeStats,
    Method,
    ErrorCode,
    MoranError,
    InvalidInputError,
    MaxStepsExceededError,
    NumericalError,
)


def fixation_probability(
        population,
        r: float,
        *,
        epsilon: float = 0.1,
        delta: float = 0.25,
        method: str = "auto",
        seed: int | None = None,
        num_threads: int = 0,
) -> FixationResult:
    """Compute fixation probability with automatic algorithm selection."""
    actual_seed = 0 if seed is None else seed

    match method:
        case "auto":
            result = exact.try_exact(population, r)
            if result is not None:
                return result
            return algorithms.naive.estimate(
                population, r, epsilon=epsilon, delta=delta,
                seed=actual_seed, num_threads=num_threads)

        case "exact":
            result = exact.try_exact(population, r)
            if result is None:
                raise ValueError("No exact formula available for this graph")
            return result

        case "naive":
            return algorithms.naive.estimate(
                population, r, epsilon=epsilon, delta=delta,
                seed=actual_seed, num_threads=num_threads)

        case _:
            raise ValueError(
                f"Unknown method {method!r}. Choose from: 'auto', 'exact', 'naive'"
            )


__all__ = [
    "__version__",
    "fixation_probability",
    "FixationResult",
    "DegreeStats",
    "Method",
    "MoranError",
    "InvalidInputError",
    "MaxStepsExceededError",
    "NumericalError",
    "exact",
    "graph",
    "algorithms",
]
