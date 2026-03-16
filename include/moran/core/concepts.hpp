#pragma once
/// @file concepts.hpp
/// @brief C++23 concepts constraining template parameters throughout the library.

#include <concepts>
#include <ranges>

#include "types.hpp"

namespace moran {

template <typename T>
concept MoranScalar = std::floating_point<T>;

/// Read-only graph interface used by all algorithms.
template <typename G>
concept Graph = requires(const G& g, VertexId v) {
    { g.num_vertices() } -> std::convertible_to<std::size_t>;
    { g.degree(v) } -> std::convertible_to<std::size_t>;

    // neighbors(v) must be a sized, random-access range of VertexId
    requires std::ranges::random_access_range<decltype(g.neighbors(v))>;
    requires std::ranges::sized_range<decltype(g.neighbors(v))>;
    requires std::convertible_to<std::ranges::range_value_t<decltype(g.neighbors(v))>, VertexId>;
};

/// Graph with precomputed degree statistics.
template <typename G>
concept DegreeStatsGraph = Graph<G> && requires(const G& g) {
    { g.degree_stats() } -> std::convertible_to<DegreeStats>;
};

}  // namespace moran
