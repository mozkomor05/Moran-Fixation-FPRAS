#pragma once
/// @file parallel.hpp
/// @brief OpenMP parallel MC driver.

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "random.hpp"

namespace moran {

/// Resolve the actual number of threads to use.
/// If requested <= 0, returns omp_get_max_threads() (or 1 without OpenMP).
[[nodiscard]] inline int resolve_num_threads(const int requested) noexcept {
    if (requested > 0) {
        return requested;
    }
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/// Compute how many samples thread `tid` should process, given `total`
/// samples spread across `num_threads` threads via round-robin.
[[nodiscard]] inline std::uint64_t samples_for_thread(const std::uint64_t total,
                                                      const int num_threads,
                                                      const int tid) noexcept {
    assert(num_threads > 0 && "samples_for_thread: num_threads must be > 0");
    if (num_threads <= 0) {
        return 0;
    }
    const auto nt = static_cast<std::uint64_t>(num_threads);
    const auto base = total / nt;
    const auto extra = (static_cast<std::uint64_t>(tid) < (total % nt)) ? 1ULL : 0ULL;
    return base + extra;
}

/// Cache-line aligned wrapper to prevent false sharing between per-thread
/// accumulators in parallel MC simulations.
template <typename T>
struct alignas(64) CacheAligned {
    T value{};
};

/// Result of a parallel MC simulation.
template <typename ThreadState>
struct ParallelMCResult {
    std::vector<CacheAligned<ThreadState>> per_thread{};
    bool aborted = false;
};

/// Distribute num_samples across threads, each with an independent RNG stream.
template <typename ThreadState, typename BodyFn>
[[nodiscard]] ParallelMCResult<ThreadState> run_parallel_mc(std::uint64_t num_samples,
                                                            std::uint64_t seed,
                                                            int num_threads_hint, BodyFn&& body) {
    // Without OpenMP, force single thread regardless of hint to avoid
    // silently underprocessing samples.
#ifdef _OPENMP
    const int num_threads = resolve_num_threads(num_threads_hint);
#else
    const int num_threads = 1;
    (void)num_threads_hint;
#endif

    auto engines = make_thread_engines(seed, num_threads);
    std::vector<CacheAligned<ThreadState>> per_thread(static_cast<std::size_t>(num_threads));
    std::atomic<bool> aborted{false};
    std::exception_ptr thread_exception;

    // Materialize callable before the parallel region.  std::forward
    // performs at most one move (for rvalue BodyFn); all threads then
    // invoke the same stable lvalue -- no moved-from UB.
    auto body_fn = std::forward<BodyFn>(body);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const auto my_samples = samples_for_thread(num_samples, num_threads, tid);
        try {
            body_fn(per_thread[tid].value, my_samples, engines[tid], aborted);
        } catch (...) {
            // Capture first exception; signal other threads to stop.
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (!thread_exception) {
                    thread_exception = std::current_exception();
                }
            }
            aborted.store(true, std::memory_order_release);
        }
    }

    if (thread_exception) {
        std::rethrow_exception(thread_exception);
    }

    return {std::move(per_thread), aborted.load(std::memory_order_acquire)};
}

}  // namespace moran
