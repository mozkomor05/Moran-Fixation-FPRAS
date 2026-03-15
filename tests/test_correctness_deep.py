"""Correctness tests: numerical stability, analytical results, MC validation."""

import math
import pytest
import moran
from moran.graph import complete_graph, cycle_graph, star_graph


class TestNumericalStability:
    def test_r_near_1(self):
        for r in [1.0 + 1e-10, 1.0 - 1e-10, 1.0 + 1e-14]:
            for n in [10, 100]:
                result = moran.exact.well_mixed(n, r)
                assert result == pytest.approx(1.0 / n, rel=0.01), \
                    f"Numerical instability at r={r}, n={n}: got {result}"

    def test_extreme_r(self):
        for r in [100.0, 0.01]:
            result = moran.exact.well_mixed(10, r)
            assert math.isfinite(result)
            assert 0.0 <= result <= 1.0


class TestAnalyticalResults:
    def test_well_mixed_formula(self):
        for N in [5, 10, 20]:
            for r in [0.5, 1.5, 2.0]:
                expected = (1.0 - 1.0/r) / (1.0 - (1.0/r)**N)
                assert moran.exact.well_mixed(N, r) == pytest.approx(expected, rel=1e-10)

    def test_monotonicity_in_r(self):
        prev = 0.0
        for r in [0.5, 1.0, 1.5, 2.0, 3.0]:
            p = moran.exact.well_mixed(20, r)
            assert p >= prev - 1e-10
            prev = p


class TestMonteCarloValidation:
    @pytest.mark.slow
    def test_naive_mc_matches_exact_on_cycle(self):
        N, r = 10, 1.5
        exact_val = moran.exact.well_mixed(N, r)
        result = moran.fixation_probability(
            cycle_graph(N), r, method="naive", epsilon=0.05, seed=42)
        assert result.estimate == pytest.approx(exact_val, abs=0.05)


class TestGraphConstruction:
    def test_complete_graph_edges(self):
        for n in [3, 5, 10]:
            g = complete_graph(n)
            assert g.num_edges() == n * (n - 1) // 2
            assert g.degree_stats().is_regular

    def test_cycle_graph_regularity(self):
        for n in [3, 5, 10]:
            assert cycle_graph(n).degree_stats().is_regular

    def test_star_graph_degrees(self):
        g = star_graph(10)
        assert g.degree(0) == 9
        for v in range(1, 10):
            assert g.degree(v) == 1

    def test_connectivity(self):
        for factory in [complete_graph, cycle_graph, star_graph]:
            assert factory(10).is_connected()


class TestAdditionalEdgeCases:
    def test_single_vertex(self):
        assert moran.exact.well_mixed(1, 2.0) == pytest.approx(1.0)
        assert moran.exact.r_equals_1(1) == pytest.approx(1.0)

    def test_r1_exactly(self):
        for n in [2, 10, 100]:
            assert moran.exact.well_mixed(n, 1.0) == pytest.approx(1.0 / n, rel=1e-10)
