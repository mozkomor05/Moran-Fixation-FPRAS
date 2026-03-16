#pragma once
/// @file graph_factories.hpp
/// @brief Factory functions for common graph topologies.

#include <cstddef>
#include <span>
#include <vector>

#include "../core/concepts.hpp"
#include "../core/types.hpp"
#include "csr_graph.hpp"

namespace moran {

/// Build a complete graph K_n (all-to-all connections, weight 1).
/// The well-mixed population is equivalent to the Moran process on K_n.
template <MoranScalar W = double>
[[nodiscard]] CSRGraph<W> make_complete_graph(const std::size_t n) {
    using Edge = CSRGraph<W>::Edge;
    std::vector<Edge> edges;
    if (n > 1) {
        edges.reserve(n * (n - 1) / 2);
    }
    for (VertexId i = 0; i < static_cast<VertexId>(n); ++i) {
        for (VertexId j = i + 1; j < static_cast<VertexId>(n); ++j) {
            edges.push_back({i, j});
        }
    }
    return CSRGraph<W>(n, std::span<const Edge>(edges));
}

/// Build a cycle graph C_n.
template <MoranScalar W = double>
[[nodiscard]] CSRGraph<W> make_cycle_graph(const std::size_t n) {
    using Edge = CSRGraph<W>::Edge;
    if (n <= 1) {
        return CSRGraph<W>(n, std::span<const Edge>{});
    }
    std::vector<Edge> edges;
    edges.reserve(n);
    for (VertexId i = 0; i < static_cast<VertexId>(n); ++i) {
        edges.push_back({i, static_cast<VertexId>((i + 1) % n)});
    }
    return CSRGraph<W>(n, std::span<const Edge>(edges));
}

/// Build a star graph S_n (vertex 0 is the hub).
/// Amplifier of selection under Bd updating (Lieberman, Hauert & Nowak, Nature 2005).
template <MoranScalar W = double>
[[nodiscard]] CSRGraph<W> make_star_graph(const std::size_t n) {
    using Edge = CSRGraph<W>::Edge;
    if (n <= 1) {
        return CSRGraph<W>(n, std::span<const Edge>{});
    }
    std::vector<Edge> edges;
    edges.reserve(n - 1);
    for (VertexId i = 1; i < static_cast<VertexId>(n); ++i) {
        edges.push_back({0, i});
    }
    return CSRGraph<W>(n, std::span<const Edge>(edges));
}

/// Build a double star graph D(a,b): two hubs connected, hub 0 has a leaves, hub 1 has b leaves.
/// Total vertices: a + b + 2. Amplifier of selection under Bd updating (Lieberman et al. 2005).
template <MoranScalar W = double>
[[nodiscard]] CSRGraph<W> make_double_star_graph(const std::size_t a, const std::size_t b) {
    using Edge = CSRGraph<W>::Edge;
    const auto n = a + b + 2;
    std::vector<Edge> edges;
    edges.reserve(a + b + 1);
    edges.push_back({0, 1});  // hub-hub edge
    for (std::size_t i = 0; i < a; ++i) {
        edges.push_back({0, static_cast<VertexId>(2 + i)});
    }
    for (std::size_t i = 0; i < b; ++i) {
        edges.push_back({1, static_cast<VertexId>(2 + a + i)});
    }
    return CSRGraph<W>(n, std::span<const Edge>(edges));
}

}  // namespace moran