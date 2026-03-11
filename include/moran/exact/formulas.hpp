#pragma once
/// @file formulas.hpp
/// @brief Exact fixation probability formulas.
///
/// Numerically stable implementations using log1p/expm1 for r near 1.

#include <cmath>
#include <cstddef>
#include <format>
#include <optional>
#include <span>
#include <vector>

#include "../core/concepts.hpp"
#include "../core/numeric.hpp"
#include "../core/result.hpp"
#include "../core/types.hpp"
#include "../graph/graph_utils.hpp"
#include "../graph/graph_validation.hpp"

namespace moran::exact {

/// Well-mixed fixation probability: (1 - 1/r) / (1 - 1/r^n).
/// For r near 1, returns 1/n via L'Hopital (handled by stable_fixation_ratio).
[[nodiscard]] inline double well_mixed(const std::size_t n, const double r) {
    if (n == 0) {
        return 0.0;
    }
    if (n == 1) {
        return 1.0;
    }
    return numeric::stable_fixation_ratio(r, 1.0, static_cast<double>(n));
}

/// Isothermal theorem: regular graphs have the same fixation probability
/// as well-mixed (Lieberman, Hauert & Nowak, Nature 2005).
[[nodiscard]] inline double isothermal_regular(const std::size_t n, const double r) {
    return well_mixed(n, r);
}

/// Neutral drift (r = 1): fixation probability is exactly 1/n.
[[nodiscard]] inline double r_equals_1(const std::size_t n) {
    if (n == 0) {
        return 0.0;
    }
    return 1.0 / static_cast<double>(n);
}

/// Fixation probability from i0 initial mutants (constant fitness r).
/// Formula: (1 - r^{-i0}) / (1 - r^{-N}).
template <MoranScalar Scalar>
[[nodiscard]] Result<Scalar> fixation_from(const std::size_t N, const std::size_t i0,
                                           const Scalar r) {
    if (auto v = validate_population_size(N); !v) {
        return std::unexpected(v.error());
    }
    if (auto v = validate_fitness(r); !v) {
        return std::unexpected(v.error());
    }
    if (i0 > N) {
        return make_error(ErrorCode::InvalidInitialState,
                          std::format("Initial mutants {} > population size {}", i0, N));
    }
    if (i0 == 0) {
        return Scalar(0);
    }
    if (i0 == N) {
        return Scalar(1);
    }
    return numeric::stable_fixation_ratio(r, static_cast<Scalar>(i0), static_cast<Scalar>(N));
}

/// Single-mutant fixation probability (i0 = 1).
template <MoranScalar Scalar>
[[nodiscard]] Result<Scalar> fixation_exact(const std::size_t N, const Scalar r) {
    return fixation_from(N, std::size_t{1}, r);
}

/// General fixation probability with state-dependent gamma ratios.
/// gamma(i) = death_rate(i) / birth_rate(i) for state i.
/// Formula: rho = 1 / (1 + sum_{k=1}^{N-1} prod_{j=1}^{k} gamma(j))
/// Computed in log-space via log-sum-exp for numerical stability.
template <MoranScalar Scalar, typename GammaFn>
    requires std::invocable<GammaFn, std::size_t> &&
             std::convertible_to<std::invoke_result_t<GammaFn, std::size_t>, Scalar>
[[nodiscard]] Result<Scalar> fixation_general(const std::size_t N, const std::size_t i0,
                                              const GammaFn& gamma) {
    if (auto v = validate_population_size(N); !v) {
        return std::unexpected(v.error());
    }
    if (i0 == 0) {
        return Scalar(0);
    }
    if (i0 >= N) {
        return Scalar(1);
    }

    std::vector<Scalar> log_gamma(N - 1);
    for (std::size_t i = 1; i < N; ++i) {
        const Scalar g = gamma(i);
        if (!(g > Scalar(0)) || !std::isfinite(g)) {
            return make_error(
                ErrorCode::InvalidFitness,
                std::format("gamma({}) must be positive, got {}", i, static_cast<double>(g)));
        }
        log_gamma[i - 1] = std::log(g);
    }

    std::vector<Scalar> log_prod(N - 1);
    log_prod[0] = log_gamma[0];
    for (std::size_t k = 1; k < N - 1; ++k) {
        log_prod[k] = log_prod[k - 1] + log_gamma[k];
    }

    std::vector<Scalar> num_terms;
    num_terms.reserve(i0);
    num_terms.push_back(Scalar(0));
    for (std::size_t k = 0; k < i0 - 1; ++k) {
        num_terms.push_back(log_prod[k]);
    }

    std::vector<Scalar> den_terms;
    den_terms.reserve(N);
    den_terms.push_back(Scalar(0));
    for (std::size_t k = 0; k < N - 1; ++k) {
        den_terms.push_back(log_prod[k]);
    }

    const Scalar log_num = numeric::log_sum_exp<Scalar>(std::span<const Scalar>(num_terms));
    const Scalar log_den = numeric::log_sum_exp<Scalar>(std::span<const Scalar>(den_terms));
    return std::exp(log_num - log_den);
}

/// Try to compute exact fixation probability for a given graph.
/// Returns std::nullopt if no exact formula applies.
///
/// Decision logic:
///   1. r == 1 (within 1e-12) -> 1/n
///   2. Graph is complete (degree = n-1) -> well-mixed formula
///   3. Graph is regular -> isothermal formula
///   4. Otherwise -> nullopt
template <typename GraphType>
[[nodiscard]] std::optional<FixationResult> try_exact(const GraphType& g, const double r) {
    const auto n = g.num_vertices();
    if (n == 0) {
        return std::nullopt;
    }

    constexpr double kRTol = 1e-12;
    if (std::abs(r - 1.0) < kRTol) {
        const double p = r_equals_1(n);
        return FixationResult{
            .estimate = p,
            .ci_lower = p,
            .ci_upper = p,
            .method = Method::exact_r_equals_1,
            .epsilon = 0.0,
            .delta = 0.0,
        };
    }

    if constexpr (DegreeStatsGraph<GraphType>) {
        const auto stats = g.degree_stats();
        if (stats.is_regular) {
            const double p = isothermal_regular(n, r);
            const auto method = (stats.max_degree == n - 1) ? Method::exact_well_mixed
                                                            : Method::exact_isothermal_regular;
            return FixationResult{
                .estimate = p,
                .ci_lower = p,
                .ci_upper = p,
                .method = method,
                .epsilon = 0.0,
                .delta = 0.0,
            };
        }
    }

    return std::nullopt;
}

}  // namespace moran::exact
