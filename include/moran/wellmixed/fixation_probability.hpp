#pragma once
/// @file fixation_probability.hpp
/// @brief Exact fixation probability for the well-mixed Moran process.

#include <cmath>
#include <cstddef>
#include <format>
#include <vector>

#include "../core/concepts.hpp"
#include "../core/numeric.hpp"
#include "../core/result.hpp"
#include "../core/types.hpp"
#include "../graph/graph_validation.hpp"

namespace moran::wellmixed {

/// Fixation probability from i0 initial mutants (constant fitness r).
template <MoranScalar Scalar>
[[nodiscard]] Result<Scalar> fixation_probability_from(std::size_t N, std::size_t i0, Scalar r) {
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
[[nodiscard]] Result<Scalar> fixation_probability_exact(std::size_t N, Scalar r) {
    return fixation_probability_from(N, std::size_t{1}, r);
}

/// General fixation probability with state-dependent gamma ratios.
template <MoranScalar Scalar, typename GammaFn>
    requires std::invocable<GammaFn, std::size_t> &&
             std::convertible_to<std::invoke_result_t<GammaFn, std::size_t>, Scalar>
[[nodiscard]] Result<Scalar> fixation_probability_general(std::size_t N, std::size_t i0,
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
        Scalar g = gamma(i);
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

}  // namespace moran::wellmixed
