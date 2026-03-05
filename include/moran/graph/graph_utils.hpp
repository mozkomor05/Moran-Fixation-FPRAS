#pragma once
/// @file graph_utils.hpp
/// @brief Graph utility algorithms and precomputation.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../core/concepts.hpp"
#include "../core/types.hpp"

namespace moran {

/// Compute degree statistics for any Graph.
template <Graph G>
[[nodiscard]] DegreeStats compute_degree_stats(const G& graph) {
    const auto n = graph.num_vertices();
    if (n == 0) {
        return {.min_degree = 0,
                .max_degree = 0,
                .avg_degree = 0.0,
                .is_regular = true,
                .num_edges = 0};
    }

    std::size_t min_d = graph.degree(0);
    std::size_t max_d = graph.degree(0);
    std::uint64_t sum_d = graph.degree(0);

    for (VertexId v = 1; v < static_cast<VertexId>(n); ++v) {
        const auto d = graph.degree(v);
        min_d = std::min(min_d, d);
        max_d = std::max(max_d, d);
        sum_d += d;
    }

    return {.min_degree = min_d,
            .max_degree = max_d,
            .avg_degree = static_cast<double>(sum_d) / static_cast<double>(n),
            .is_regular = (min_d == max_d),
            .num_edges = sum_d / 2};
}

/// Check if a graph is isothermal (equal weighted degree at all vertices).
template <Graph G>
    requires requires(const G& g, VertexId v) {
        { g.weighted_degree(v) } -> std::convertible_to<double>;
    }
[[nodiscard]] bool is_isothermal(const G& graph, const double tolerance = 1e-10) {
    const auto n = graph.num_vertices();
    if (n <= 1) {
        return true;
    }

    const auto ref = static_cast<double>(graph.weighted_degree(0));
    for (VertexId v = 1; v < static_cast<VertexId>(n); ++v) {
        if (std::abs(static_cast<double>(graph.weighted_degree(v)) - ref) > tolerance) {
            return false;
        }
    }
    return true;
}

namespace detail {

/// Precompute 1/degree(v) for all vertices. Shared read-only across threads.
template <Graph GraphType>
[[nodiscard]] std::vector<double> compute_inv_degree(const GraphType& graph) {
    const auto n = graph.num_vertices();
    std::vector<double> inv_degree(n);
    for (VertexId v = 0; v < static_cast<VertexId>(n); ++v) {
        const auto d = graph.degree(v);
        assert(d > 0 && "compute_inv_degree requires no isolated vertices");
        inv_degree[v] = 1.0 / static_cast<double>(d);
    }
    return inv_degree;
}

}  // namespace detail

}  // namespace moran
