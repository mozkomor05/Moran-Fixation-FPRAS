"""Validation tests for the Moran process library."""

import math
import pytest
import moran
from moran.graph import CSRGraph, complete_graph, cycle_graph, star_graph


class TestExactFormulas:
    def test_neutral_drift(self):
        for n in [2, 5, 10, 50, 100]:
            assert moran.exact.r_equals_1(n) == pytest.approx(1.0 / n, abs=1e-12)

    def test_well_mixed_advantaged(self):
        expected = (1.0 - 0.5) / (1.0 - 0.5**10)
        assert moran.exact.well_mixed(10, 2.0) == pytest.approx(expected, abs=1e-12)

    def test_well_mixed_disadvantaged(self):
        expected = (1.0 - 2.0) / (1.0 - 2.0**10)
        assert moran.exact.well_mixed(10, 0.5) == pytest.approx(expected, abs=1e-12)

    def test_isothermal_matches_well_mixed(self):
        assert moran.exact.isothermal_regular(10, 1.5) == pytest.approx(
            moran.exact.well_mixed(10, 1.5), abs=1e-12)

    def test_try_exact_regular(self):
        g = cycle_graph(10)
        result = moran.exact.try_exact(g, 1.5)
        assert result is not None
        assert result.method == moran.Method.exact_isothermal_regular

    def test_try_exact_r1(self):
        g = star_graph(10)
        result = moran.exact.try_exact(g, 1.0)
        assert result is not None
        assert result.method == moran.Method.exact_r_equals_1
        assert result.estimate == pytest.approx(0.1, abs=1e-12)

    def test_try_exact_non_regular(self):
        g = star_graph(10)
        assert moran.exact.try_exact(g, 1.5) is None


class TestFixationProbability:
    def test_auto_exact_well_mixed(self):
        result = moran.fixation_probability(complete_graph(10), 1.5)
        assert result.method == moran.Method.exact_well_mixed

    def test_auto_exact_isothermal(self):
        result = moran.fixation_probability(cycle_graph(10), 1.5)
        assert result.method == moran.Method.exact_isothermal_regular

    def test_auto_exact_r1(self):
        result = moran.fixation_probability(star_graph(10), 1.0)
        assert result.method == moran.Method.exact_r_equals_1

    def test_auto_fallback_to_mc(self):
        result = moran.fixation_probability(star_graph(10), 1.5, seed=42)
        assert result.method in (moran.Method.mc_naive, moran.Method.fpras_goldberg)
        assert result.samples > 0

    def test_method_exact(self):
        result = moran.fixation_probability(cycle_graph(10), 1.5, method="exact")
        assert result.method == moran.Method.exact_isothermal_regular

    def test_method_exact_fails_for_non_regular(self):
        with pytest.raises(ValueError, match="No exact formula"):
            moran.fixation_probability(star_graph(10), 1.5, method="exact")

    def test_method_naive(self):
        result = moran.fixation_probability(complete_graph(8), 1.5, method="naive", seed=42)
        assert result.method == moran.Method.mc_naive


class TestFixationResult:
    def test_result_fields(self):
        result = moran.fixation_probability(cycle_graph(10), 1.5)
        assert hasattr(result, "estimate")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "method")
        assert hasattr(result, "epsilon")
        assert hasattr(result, "delta")
        assert hasattr(result, "samples")
        assert hasattr(result, "elapsed_seconds")

    def test_to_dict(self):
        d = moran.fixation_probability(cycle_graph(10), 1.5).to_dict()
        assert "estimate" in d
        assert "method" in d

    def test_repr(self):
        assert "FixationResult" in repr(moran.fixation_probability(cycle_graph(10), 1.5))


class TestGraphConstruction:
    def test_complete_graph(self):
        g = complete_graph(5)
        assert g.num_vertices() == 5
        assert g.num_edges() == 10

    def test_cycle_graph(self):
        g = cycle_graph(6)
        assert g.num_vertices() == 6
        assert g.num_edges() == 6

    def test_star_graph(self):
        g = star_graph(5)
        assert g.num_vertices() == 5
        assert g.degree(0) == 4

    def test_degree_stats(self):
        stats = star_graph(5).degree_stats()
        assert stats.min_degree == 1
        assert stats.max_degree == 4
        assert not stats.is_regular

    def test_from_numpy(self):
        import numpy as np
        g = CSRGraph(3, np.array([0, 1, 2], dtype=np.uint32),
                        np.array([1, 2, 0], dtype=np.uint32))
        assert g.num_vertices() == 3


class TestEdgeCases:
    def test_invalid_fitness(self):
        from moran.algorithms import naive
        with pytest.raises(moran.InvalidInputError):
            naive.estimate(star_graph(5), 0.0)

    def test_nan_fitness(self):
        from moran.algorithms import naive
        with pytest.raises(moran.InvalidInputError):
            naive.estimate(star_graph(5), float("nan"))

    def test_disconnected_graph(self):
        import numpy as np
        from moran.algorithms import naive
        g = CSRGraph(4, np.array([0, 2], dtype=np.uint32),
                        np.array([1, 3], dtype=np.uint32))
        with pytest.raises(moran.InvalidInputError):
            naive.estimate(g, 1.5)
