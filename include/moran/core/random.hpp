#pragma once
/// @file random.hpp
/// @brief Xoshiro256** PRNG and MC sampling utilities.
///
/// Xoshiro256** (Blackman & Vigna 2018). 2^128 jump for parallel streams.

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "concepts.hpp"

namespace moran {

/// Xoshiro256** PRNG (Blackman & Vigna 2018).
class Xoshiro256StarStar {
public:
    using result_type = std::uint64_t;

    /// Seed via SplitMix64 (Vigna's recommended initialization).
    explicit Xoshiro256StarStar(const std::uint64_t seed = 0) noexcept {
        auto splitmix = [z = seed]() mutable -> std::uint64_t {
            z += 0x9e3779b97f4a7c15ULL;
            std::uint64_t result = z;
            result = (result ^ (result >> 30)) * 0xbf58476d1ce4e5b9ULL;
            result = (result ^ (result >> 27)) * 0x94d049bb133111ebULL;
            return result ^ (result >> 31);
        };
        for (auto& s : state_) {
            s = splitmix();
        }
    }

    static constexpr result_type min() noexcept { return 0; }
    static constexpr result_type max() noexcept {
        return std::numeric_limits<result_type>::max();
    }

    result_type operator()() noexcept {
        const std::uint64_t result = std::rotl(state_[1] * 5, 7) * 9;
        const std::uint64_t t = state_[1] << 17;  // NOLINT(cppcoreguidelines-init-variables)

        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = std::rotl(state_[3], 45);

        return result;
    }

    /// Jump forward by 2^128 steps for parallel stream independence.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    void jump() noexcept {
        static constexpr std::array<std::uint64_t, 4> kJump = {
            0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
            0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL
        };
        std::array<std::uint64_t, 4> s = {};
        for (const auto jmp : kJump) {
            for (int b = 0; b < 64; ++b) {
                if ((jmp & (std::uint64_t{1} << b)) != 0) {
                    for (int k = 0; k < 4; ++k) {
                        s[k] ^= state_[k];
                    }
                }
                (*this)();
            }
        }
        state_ = s;
    }

private:
    std::array<std::uint64_t, 4> state_{};
};

/// Create independent RNG engines via 2^128 jumps.
inline std::vector<Xoshiro256StarStar> make_thread_engines(
    const std::uint64_t master_seed,
    const int num_threads)
{
    std::vector<Xoshiro256StarStar> engines;
    engines.reserve(static_cast<std::size_t>(num_threads));

    Xoshiro256StarStar master(master_seed);
    for (int i = 0; i < num_threads; ++i) {
        engines.push_back(master);
        master.jump();
    }
    return engines;
}

/// Generate a uniform real in [0, 1).
template <MoranScalar Scalar>
[[nodiscard]] Scalar uniform_01(Xoshiro256StarStar& rng) noexcept {
    constexpr Scalar scale = Scalar(1.0) / Scalar(1ULL << 53);
    return static_cast<Scalar>(rng() >> 11) * scale;
}

/// Sample from Geometric(p) via inverse-CDF. Saturates to prevent overflow.
template <MoranScalar Scalar>
[[nodiscard]] std::uint64_t geometric_sample(
    const Scalar p,
    Xoshiro256StarStar& rng) noexcept
{
    constexpr std::uint64_t kMaxGeometric = std::numeric_limits<std::uint64_t>::max() / 2;
    if (p >= Scalar(1)) {
        return 0;
    }
    if (p <= Scalar(0)) {
        return kMaxGeometric;
    }

    const Scalar u = uniform_01<Scalar>(rng);
    if (u == Scalar(0)) {
        return kMaxGeometric;
    }

    const Scalar log1mp = std::log1p(-p);
    const auto k = std::log(u) / log1mp;

    constexpr auto max_safe =
        static_cast<Scalar>(kMaxGeometric);
    if (!(k < max_safe)) {  // also catches NaN
        return kMaxGeometric;
    }
    return static_cast<std::uint64_t>(k);
}

/// Uniform random index in [0, n) with FP safety clamp.
[[nodiscard]] inline std::size_t uniform_index(Xoshiro256StarStar& rng, const std::size_t n) noexcept {
    assert(n > 0 && "uniform_index: n must be > 0");
    auto idx = static_cast<std::size_t>(
        uniform_01<double>(rng) * static_cast<double>(n));
    if (idx >= n) {
        idx = n - 1;
    }
    return idx;
}

/// Resolve RNG seed: 0 means draw from std::random_device.
[[nodiscard]] inline std::uint64_t resolve_seed(const std::uint64_t seed) {
    if (seed != 0) { return seed; }
    std::random_device rd;
    return (static_cast<std::uint64_t>(rd()) << 32) | rd();
}

} // namespace moran
