#pragma once
/// @file params.hpp
/// @brief FPRAS parameter derivation from epsilon/delta.
///
/// Formulas from:
///   - Diaz et al. (Algorithmica 2014), Theorem 13 / Corollary 10

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>

#include "../core/result.hpp"
#include "../core/types.hpp"

namespace moran::fpras {

/// Derived simulation parameters (sample count + step limit per run).
struct DerivedParams {
    std::uint64_t samples;
    std::uint64_t per_run_step_limit;
};

/// Validate accuracy parameters at runtime.
[[nodiscard]] inline Result<void> validate_accuracy(const Accuracy& acc) {
    if (!(acc.epsilon > 0.0) || !(acc.epsilon < 1.0) || !std::isfinite(acc.epsilon)) {
        return make_error(ErrorCode::InvalidConfig,
                          std::format("epsilon must be in (0, 1), got {}", acc.epsilon));
    }
    if (!(acc.delta > 0.0) || !(acc.delta < 1.0) || !std::isfinite(acc.delta)) {
        return make_error(ErrorCode::InvalidConfig,
                          std::format("delta must be in (0, 1), got {}", acc.delta));
    }
    return {};
}

/// Safe ceiling of a double to uint64_t, clamping to a sane max.
[[nodiscard]] inline std::uint64_t safe_ceil(const double x) noexcept {
    constexpr auto kMax = static_cast<double>(std::uint64_t{1} << 62);
    if (!std::isfinite(x) || x <= 0.0) {
        return 1;
    }
    if (x >= kMax) {
        return std::uint64_t{1} << 62;
    }
    return static_cast<std::uint64_t>(std::ceil(x));
}

/// Median boosting factor: ceil(log_4(1/delta)).
/// Standard Chernoff-based amplification (Jerrum, Valiant & Vazirani 1986).
[[nodiscard]] inline std::uint64_t median_boost(const double delta) noexcept {
    if (delta >= 0.25) {
        return 1;
    }
    const double ratio = std::log2(1.0 / delta) / std::log2(4.0);
    return std::max(std::uint64_t{1}, safe_ceil(ratio));
}

/// Diaz absorption time bound (Corollary 10).
/// E[tau] <= r/|r-1| * n^4, with safety_factor=4 for Markov's inequality.
[[nodiscard]] inline double diaz_absorption_bound(const std::size_t n, const double r) noexcept {
    constexpr double safety_factor = 4.0;
    const double nd = static_cast<double>(n);
    const double n4 = nd * nd * nd * nd;
    const double diff = std::abs(r - 1.0);
    const double factor = (diff >= 1e-9) ? (r / diff) : 1.0;
    return safety_factor * factor * n4;
}

/// Diaz 2014 naive MC parameters (Theorem 13).
/// N = ceil(0.5 * eps^{-2} * n^2 * ln(16)).
[[nodiscard]] inline DerivedParams diaz_naive(const std::size_t n, const double r,
                                              const Accuracy& acc) {
    const double nd = static_cast<double>(n);
    const double eps = acc.epsilon;
    const double z = 0.5 * nd * nd * std::log(16.0) / (eps * eps);
    const auto boost = median_boost(acc.delta);
    const auto samples = safe_ceil(z) * boost;
    const auto u = safe_ceil(diaz_absorption_bound(n, r));
    return {.samples = samples, .per_run_step_limit = u};
}

/// Well-mixed MC parameters (Hoeffding bound for Bernoulli trials).
/// Same sample count as Diaz naive. No step limit (geometric sampling).
[[nodiscard]] inline DerivedParams well_mixed_mc(const std::size_t N, const Accuracy& acc) {
    const double nd = static_cast<double>(N);
    const double eps = acc.epsilon;
    const double z = 0.5 * nd * nd * std::log(16.0) / (eps * eps);
    const auto boost = median_boost(acc.delta);
    const auto samples = safe_ceil(z) * boost;
    return {.samples = samples, .per_run_step_limit = 0};
}

/// Multiplicative CI: [est/(1+eps), est/(1-eps)], clamped to [0, 1].
[[nodiscard]] inline std::pair<double, double> multiplicative_ci(const double estimate,
                                                                 const double epsilon) noexcept {
    return {std::max(0.0, estimate / (1.0 + epsilon)), std::min(1.0, estimate / (1.0 - epsilon))};
}

}  // namespace moran::fpras
