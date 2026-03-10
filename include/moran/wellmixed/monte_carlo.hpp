#pragma once
/// @file monte_carlo.hpp
/// @brief Monte Carlo simulation of the well-mixed Moran process.
///
/// Geometric sampling skips identity steps in O(1).

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>

#include "../core/concepts.hpp"
#include "../core/parallel.hpp"
#include "../core/random.hpp"
#include "../core/result.hpp"
#include "../core/types.hpp"
#include "../fpras/params.hpp"
#include "../graph/graph_validation.hpp"

namespace moran::wellmixed {

namespace detail {

/// Single well-mixed Moran process run. Returns true on fixation.
template <MoranScalar Scalar>
[[nodiscard]] bool simulate_single_run(std::size_t N, Scalar r, std::size_t i0,
                                       Xoshiro256StarStar& rng, std::uint64_t max_steps) {
    auto i = i0;
    const auto Ns = static_cast<Scalar>(N);
    std::uint64_t steps = 0;

    while (i > 0 && i < N) {
        if (max_steps > 0 && steps >= max_steps) {
            break;
        }

        const auto is = static_cast<Scalar>(i);
        const Scalar ri = r * is;
        const Scalar nmi = Ns - is;
        const Scalar total_fitness = ri + nmi;

        const Scalar t_plus = (ri / total_fitness) * (nmi / Ns);
        const Scalar t_minus = (nmi / total_fitness) * (is / Ns);
        const Scalar p_active = t_plus + t_minus;
        if (!(p_active > Scalar(0))) {
            break;
        }

        // Geometric sampling: skip identity steps in O(1).
        const auto skipped = geometric_sample<Scalar>(p_active, rng);
        constexpr auto kMaxSteps = std::numeric_limits<std::uint64_t>::max() / 2;
        steps = (skipped > kMaxSteps - steps) ? kMaxSteps : steps + skipped;
        if (max_steps > 0 && steps >= max_steps) {
            break;
        }

        const Scalar u = uniform_01<Scalar>(rng);
        const Scalar threshold = t_plus / p_active;

        if (u < threshold) {
            ++i;
        } else {
            --i;
        }

        ++steps;
    }

    return i == N;
}

}  // namespace detail

/// Parallel MC fixation probability estimator.
/// Derives sample count from epsilon/delta via Hoeffding bound.
[[nodiscard]] inline Result<FixationResult> monte_carlo_fixation(
    std::size_t N, double r, std::size_t i0, const SimulationConfig& config = {}) {
    if (auto v = validate_population_size(N); !v) {
        return std::unexpected(v.error());
    }
    if (auto v = validate_fitness(r); !v) {
        return std::unexpected(v.error());
    }
    if (i0 == 0) {
        return FixationResult{.estimate = 0.0,
                              .ci_lower = 0.0,
                              .ci_upper = 0.0,
                              .method = Method::mc_naive,
                              .error_model = ErrorModel::additive,
                              .epsilon = config.accuracy.epsilon,
                              .delta = config.accuracy.delta};
    }
    if (i0 >= N) {
        return FixationResult{.estimate = 1.0,
                              .ci_lower = 1.0,
                              .ci_upper = 1.0,
                              .method = Method::mc_naive,
                              .error_model = ErrorModel::additive,
                              .epsilon = config.accuracy.epsilon,
                              .delta = config.accuracy.delta};
    }

    const auto seed = resolve_seed(config.seed);
    const auto params = fpras::well_mixed_mc(N, config.accuracy);
    const auto start = std::chrono::high_resolution_clock::now();

    auto mc = run_parallel_mc<std::uint64_t>(
        params.samples, seed, config.num_threads,
        [&](std::uint64_t& fixations, std::uint64_t my_samples, Xoshiro256StarStar& rng,
            [[maybe_unused]] std::atomic<bool>&) {
            for (std::uint64_t s = 0; s < my_samples; ++s) {
                if (detail::simulate_single_run<double>(N, r, i0, rng, 0)) {
                    ++fixations;
                }
            }
        });

    std::uint64_t total_fixations = 0;
    for (const auto& f : mc.per_thread) {
        total_fixations += f.value;
    }

    const double p_hat = static_cast<double>(total_fixations) / static_cast<double>(params.samples);

    const auto [ci_lo, ci_hi] = fpras::multiplicative_ci(p_hat, config.accuracy.epsilon);

    const auto end = std::chrono::high_resolution_clock::now();

    return FixationResult{.estimate = p_hat,
                          .ci_lower = ci_lo,
                          .ci_upper = ci_hi,
                          .method = Method::mc_naive,
                          .error_model = ErrorModel::multiplicative,
                          .epsilon = config.accuracy.epsilon,
                          .delta = config.accuracy.delta,
                          .samples = params.samples,
                          .elapsed_seconds = std::chrono::duration<double>(end - start).count(),
                          .seed_used = seed};
}

}  // namespace moran::wellmixed
