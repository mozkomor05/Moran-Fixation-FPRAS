#pragma once
/// @file graph_validation.hpp
/// @brief Graph and parameter validation for MC simulations.

#include <cmath>
#include <cstdint>
#include <format>
#include <queue>
#include <vector>

#include "../core/concepts.hpp"
#include "../core/result.hpp"
#include "../core/types.hpp"

namespace moran {

/// Check if a graph is connected via BFS.
template <Graph G>
[[nodiscard]] bool is_connected(const G& graph) {
    const auto n = graph.num_vertices();
    if (n <= 1) {
        return true;
    }

    std::vector<std::uint8_t> visited(n, 0);
    std::queue<VertexId> queue;
    queue.push(0);
    visited[0] = true;
    std::size_t count = 1;

    while (!queue.empty()) {
        const auto v = queue.front();
        queue.pop();
        for (const auto u : graph.neighbors(v)) {
            if (!visited[u]) {
                visited[u] = true;
                ++count;
                queue.push(u);
            }
        }
    }
    return count == n;
}

/// Validate that a fitness value is positive and finite.
template <MoranScalar Scalar>
[[nodiscard]] Result<void> validate_fitness(Scalar r) {
    if (!(r > Scalar(0)) || !std::isfinite(r)) {
        return make_error(
            ErrorCode::InvalidFitness,
            std::format("Fitness r must be finite and positive, got {}", static_cast<double>(r)));
    }
    return {};
}

/// Validate that a population size is >= 1.
[[nodiscard]] inline Result<void> validate_population_size(std::size_t N) {
    if (N == 0) {
        return make_error(ErrorCode::InvalidPopulationSize, "Population size must be >= 1");
    }
    return {};
}

/// Validate graph for MC: connected, no isolates.
template <Graph G>
[[nodiscard]] Result<void> validate_for_fpras(const G& graph) {
    if (graph.num_vertices() == 0) {
        return make_error(ErrorCode::InvalidGraph, "Graph has no vertices");
    }

    for (VertexId v = 0; v < static_cast<VertexId>(graph.num_vertices()); ++v) {
        if (graph.degree(v) == 0) {
            return make_error(ErrorCode::InvalidGraph,
                              std::format("Vertex {} has degree 0 (isolated)", v));
        }
    }

    if (!is_connected(graph)) {
        return make_error(ErrorCode::InvalidGraph, "Graph is not connected");
    }

    return {};
}

/// Validate graph + fitness for MC simulations.
template <Graph G, MoranScalar Scalar>
[[nodiscard]] Result<void> validate_graph_mc(const G& graph, Scalar r) {
    if (auto validation = validate_for_fpras(graph); !validation) {
        return std::unexpected(validation.error());
    }
    if (auto vr = validate_fitness(r); !vr) {
        return std::unexpected(vr.error());
    }
    return {};
}

}  // namespace moran
