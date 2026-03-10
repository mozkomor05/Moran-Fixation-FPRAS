#pragma once
/// @file types.hpp
/// @brief Shared types for graph-structured Moran process algorithms.

#include <cstdint>

namespace moran::graph_structured {

/// Result of a single naive Moran simulation.
struct SingleRunResult {
    bool fixation;
    bool truncated;
    std::uint64_t total_steps;
    std::uint64_t effective_steps;
};

}  // namespace moran::graph_structured
