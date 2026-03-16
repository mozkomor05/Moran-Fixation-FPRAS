#pragma once
/// @file types.hpp
/// @brief Fundamental type aliases, config structs, and result types.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace moran {

using VertexId = std::uint32_t;
using EdgeIdx = std::uint64_t;

inline constexpr VertexId kInvalidVertex = std::numeric_limits<VertexId>::max();

/// Multiplicative error tolerance and failure probability.
/// Per FPRAS convention (Diaz 2014, Chatterjee 2017):
///   P[|estimate - true| > epsilon * true] <= delta
struct Accuracy {
    double epsilon = 0.1;  ///< Multiplicative error tolerance
    double delta = 0.25;   ///< Failure probability (3/4 success default)
};

/// Unified configuration for all simulation algorithms.
/// Sample counts and step limits are derived internally from epsilon/delta.
struct SimulationConfig {
    Accuracy accuracy;
    std::uint64_t seed = 0;  ///< 0 = random from std::random_device
    int num_threads = 0;     ///< 0 = all cores
};

/// Algorithm used to compute the fixation probability.
enum class Method : std::uint8_t {
    exact_well_mixed,
    exact_r_equals_1,
    exact_isothermal_regular,
    mc_naive,
    fpras_chatterjee,
    fpras_goldberg
};

/// Result of a fixation probability computation (exact or Monte Carlo).
struct FixationResult {
    double estimate = 0.0;
    double ci_lower = 0.0;
    double ci_upper = 0.0;

    Method method = Method::mc_naive;
    double epsilon = 0.0;
    double delta = 0.0;

    std::uint64_t samples = 0;
    std::uint64_t steps_total = 0;
    std::uint64_t steps_effective = 0;
    std::uint64_t runs_aborted = 0;
    double elapsed_seconds = 0.0;
    std::uint64_t seed_used = 0;
};

struct DegreeStats {
    std::size_t min_degree;
    std::size_t max_degree;
    double avg_degree;
    bool is_regular;
    std::size_t num_edges;
};

[[nodiscard]] constexpr std::string_view method_name(const Method m) noexcept {
    switch (m) {
        case Method::exact_well_mixed:
            return "exact_well_mixed";
        case Method::exact_r_equals_1:
            return "exact_r_equals_1";
        case Method::exact_isothermal_regular:
            return "exact_isothermal_regular";
        case Method::mc_naive:
            return "mc_naive";
        case Method::fpras_chatterjee:
            return "fpras_chatterjee";
        case Method::fpras_goldberg:
            return "fpras_goldberg";
    }
    return "unknown";
}

}  // namespace moran
