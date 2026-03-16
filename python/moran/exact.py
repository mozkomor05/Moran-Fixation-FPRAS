"""Exact fixation probability formulas."""

from ._moran import FixationResult, algorithms as _alg


def well_mixed(n: int, r: float) -> float:
    """Fixation on K_n: (1-1/r) / (1-1/r^n). For r=1, returns 1/n."""
    return _alg.exact.well_mixed(n, r)


def isothermal_regular(n: int, r: float) -> float:
    """Fixation on regular graphs (isothermal theorem, Lieberman et al. 2005)."""
    return _alg.exact.isothermal_regular(n, r)


def r_equals_1(n: int) -> float:
    return _alg.exact.r_equals_1(n)


def try_exact(graph, r: float) -> FixationResult | None:
    return _alg.exact.try_exact(graph, r)


__all__ = ["well_mixed", "isothermal_regular", "r_equals_1", "try_exact"]
