#pragma once
/// @file params.hpp
/// @brief FPRAS parameter derivation from epsilon/delta.
///
/// Formulas from:
///   - Diaz et al. (Algorithmica 2014), Theorem 13 / Corollary 8 / 10

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>

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

/// Per-run step limit: 8N * E[tau], Markov/union bound (Theorem 13).
/// r > 1: E[tau] <= r/(r-1) * n^4 (Cor. 10). r < 1: E[tau] <= 1/(1-r) * n^3 (Cor. 8).
[[nodiscard]] inline double diaz_step_limit(const std::size_t n, const double r,
                                            const std::uint64_t num_samples) noexcept {
    const auto nd = static_cast<double>(n);
    const auto ns = static_cast<double>(num_samples);
    const double markov_factor = 8.0 * ns;

    const double diff = std::abs(r - 1.0);
    if (diff < 1e-9) {
        // r = 1: E[tau] <= n^6 (Theorem 11, conservative)
        return markov_factor * nd * nd * nd * nd * nd * nd;
    }
    if (r > 1.0) {
        // Cor. 10
        return markov_factor * (r / diff) * nd * nd * nd * nd;
    }
    // Cor. 8
    return markov_factor * (1.0 / diff) * nd * nd * nd;
}

/// Diaz 2014 naive MC parameters (Theorem 13).
/// N = ceil(0.5 * eps^{-2} * n^2 * ln(16)), boosted by ceil(log_4(1/delta)).
/// T = 8N * E[tau] per Markov/union bound.
[[nodiscard]] inline DerivedParams diaz_naive(const std::size_t n, const double r,
                                              const Accuracy& acc) {
    const auto nd = static_cast<double>(n);
    const auto eps = acc.epsilon;
    const double z = 0.5 * nd * nd * std::log(16.0) / (eps * eps);
    const auto boost = median_boost(acc.delta);
    const auto samples = safe_ceil(z) * boost;
    const auto step_limit = safe_ceil(diaz_step_limit(n, r, samples));
    return {.samples = samples, .per_run_step_limit = step_limit};
}

/// Multiplicative CI: [est/(1+eps), est/(1-eps)], clamped to [0, 1].
[[nodiscard]] inline std::pair<double, double> multiplicative_ci(const double estimate,
                                                                 const double epsilon) noexcept {
    return {std::max(0.0, estimate / (1.0 + epsilon)), std::min(1.0, estimate / (1.0 - epsilon))};
}

}  // namespace moran::fpras
