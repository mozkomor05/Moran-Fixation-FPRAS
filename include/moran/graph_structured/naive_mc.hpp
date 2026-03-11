#pragma once
/// @file naive_mc.hpp
/// @brief Naive Monte Carlo fixation probability on graphs.
///
/// Simulates all steps including ineffective ones (Díaz et al. 2014, §3).

#include <cassert>
#include <chrono>
#include <cstdint>
#include <vector>

#include "../core/concepts.hpp"
#include "../core/parallel.hpp"
#include "../core/random.hpp"
#include "../core/result.hpp"
#include "../core/types.hpp"
#include "../fpras/params.hpp"
#include "../graph/graph_validation.hpp"
#include "types.hpp"

namespace moran::graph_structured {

template <Graph GraphType>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] SingleRunResult simulate_naive_single(const GraphType& graph, const double r,
                                                    const VertexId initial_mutant,
                                                    Xoshiro256StarStar& rng,
                                                    const std::uint64_t max_steps) {
    const auto n = graph.num_vertices();
    assert(initial_mutant < static_cast<VertexId>(n) && "initial_mutant out of bounds");

    std::vector<std::uint8_t> is_mutant(n, 0);
    is_mutant[initial_mutant] = 1;
    std::size_t mutant_count = 1;

    std::uint64_t total_steps = 0;
    std::uint64_t effective_steps = 0;

    while (mutant_count > 0 && mutant_count < n) {
        if (max_steps > 0 && total_steps >= max_steps) {
            return {.fixation = false,
                    .truncated = true,
                    .total_steps = total_steps,
                    .effective_steps = effective_steps};
        }
        ++total_steps;

        // Pick reproducer proportional to fitness.
        // Decide type (mutant vs resident) in O(1), then scan for vertex in O(n).
        const double r_times_i = r * static_cast<double>(mutant_count);
        const double total_fitness = r_times_i + static_cast<double>(n - mutant_count);
        const double u_repr = uniform_01<double>(rng) * total_fitness;

        VertexId reproducer = 0;
        if (u_repr < r_times_i) {
            // Mutant reproducer: pick uniformly among mutants
            auto target_idx = static_cast<std::size_t>(uniform_01<double>(rng) *
                                                       static_cast<double>(mutant_count));
            if (target_idx >= mutant_count) {
                target_idx = mutant_count - 1;
            }
            std::size_t count = 0;
            for (VertexId v = 0; v < static_cast<VertexId>(n); ++v) {
                if (is_mutant[v]) {
                    if (count == target_idx) {
                        reproducer = v;
                        break;
                    }
                    ++count;
                }
            }
        } else {
            // Resident reproducer: pick uniformly among residents
            const auto res_count = n - mutant_count;
            auto target_idx =
                static_cast<std::size_t>(uniform_01<double>(rng) * static_cast<double>(res_count));
            if (target_idx >= res_count) {
                target_idx = res_count - 1;
            }
            std::size_t count = 0;
            for (VertexId v = 0; v < static_cast<VertexId>(n); ++v) {
                if (!is_mutant[v]) {
                    if (count == target_idx) {
                        reproducer = v;
                        break;
                    }
                    ++count;
                }
            }
        }

        const auto nbrs = graph.neighbors(reproducer);
        const auto deg = nbrs.size();
        if (deg == 0) {
            continue;
        }

        const VertexId replaced = nbrs[uniform_index(rng, deg)];

        if (is_mutant[replaced] != is_mutant[reproducer]) {
            ++effective_steps;
            if (is_mutant[reproducer]) {
                is_mutant[replaced] = 1;
                ++mutant_count;
            } else {
                is_mutant[replaced] = 0;
                --mutant_count;
            }
        }
    }

    return {.fixation = (mutant_count == n),
            .truncated = false,
            .total_steps = total_steps,
            .effective_steps = effective_steps};
}

namespace detail {
struct NaiveAccum {
    std::uint64_t fixations = 0;
    std::uint64_t total_steps = 0;
    std::uint64_t effective_steps = 0;
    std::uint64_t truncated = 0;
};
}  // namespace detail

/// Naive MC fixation probability (Díaz et al. 2014, §3).
/// Derives sample count from epsilon/delta.
template <Graph GraphType>
[[nodiscard]] Result<FixationResult> naive_mc_fixation(const GraphType& graph, const double r,
                                                       const SimulationConfig& config = {}) {
    if (auto v = validate_graph_mc(graph, r); !v) {
        return std::unexpected(v.error());
    }

    const auto n = graph.num_vertices();
    const auto seed = resolve_seed(config.seed);
    const auto params = fpras::diaz_naive(n, r, config.accuracy);
    const auto start_time = std::chrono::high_resolution_clock::now();

    auto mc = run_parallel_mc<detail::NaiveAccum>(
        params.samples, seed, config.num_threads,
        [&](detail::NaiveAccum& accum, std::uint64_t my_samples, Xoshiro256StarStar& rng,
            [[maybe_unused]] std::atomic<bool>&) {
            for (std::uint64_t s = 0; s < my_samples; ++s) {
                const auto initial_mutant = static_cast<VertexId>(uniform_index(rng, n));

                auto result =
                    simulate_naive_single(graph, r, initial_mutant, rng, params.per_run_step_limit);

                if (result.truncated) {
                    ++accum.truncated;
                } else if (result.fixation) {
                    ++accum.fixations;
                }
                accum.total_steps += result.total_steps;
                accum.effective_steps += result.effective_steps;
            }
        });

    std::uint64_t total_fix = 0;
    std::uint64_t total_tot = 0;
    std::uint64_t total_eff = 0;
    std::uint64_t total_truncated = 0;
    for (const auto& accum : mc.per_thread) {
        total_fix += accum.value.fixations;
        total_tot += accum.value.total_steps;
        total_eff += accum.value.effective_steps;
        total_truncated += accum.value.truncated;
    }

    if (total_truncated >= params.samples) {
        return make_error(ErrorCode::MaxStepsExceeded,
                          "All simulation runs were truncated by step limit");
    }
    const auto effective_samples = params.samples - total_truncated;

    const double p_hat = static_cast<double>(total_fix) / static_cast<double>(effective_samples);

    const auto [ci_lo, ci_hi] = fpras::multiplicative_ci(p_hat, config.accuracy.epsilon);
    const auto end_time = std::chrono::high_resolution_clock::now();

    return FixationResult{
        .estimate = p_hat,
        .ci_lower = ci_lo,
        .ci_upper = ci_hi,
        .method = Method::mc_naive,
        .epsilon = config.accuracy.epsilon,
        .delta = config.accuracy.delta,
        .samples = effective_samples,
        .steps_total = total_tot,
        .steps_effective = total_eff,
        .runs_aborted = total_truncated,
        .elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count(),
        .seed_used = seed};
}

}  // namespace moran::graph_structured
