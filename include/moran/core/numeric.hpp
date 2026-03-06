#pragma once
/// @file numeric.hpp
/// @brief Numerically stable primitives: Kahan summation, log-sum-exp,
///        stable ratio computation.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <span>
#include <vector>

#include "concepts.hpp"

namespace moran::numeric {

/// Kahan-Babuskova-Neumaier compensated summation.
template <MoranScalar Scalar>
class KahanAccumulator {
public:
    constexpr KahanAccumulator() noexcept = default;

    constexpr explicit KahanAccumulator(Scalar initial) noexcept : sum_(initial) {}

    constexpr KahanAccumulator& operator+=(Scalar value) noexcept {
        const Scalar t = sum_ + value;
        if (std::abs(sum_) >= std::abs(value)) {
            compensation_ += (sum_ - t) + value;
        } else {
            compensation_ += (value - t) + sum_;
        }
        sum_ = t;
        return *this;
    }

    [[nodiscard]] constexpr Scalar sum() const noexcept { return sum_ + compensation_; }

    constexpr void reset() noexcept {
        sum_ = Scalar(0);
        compensation_ = Scalar(0);
    }

private:
    Scalar sum_ = Scalar(0);
    Scalar compensation_ = Scalar(0);
};

/// Convenience: Kahan-sum a contiguous range.
template <MoranScalar Scalar>
[[nodiscard]] constexpr Scalar kahan_sum(std::span<const Scalar> values) noexcept {
    KahanAccumulator<Scalar> acc;
    for (const auto& v : values) {
        acc += v;
    }
    return acc.sum();
}

/// Numerically stable log(sum exp(x_i)) via max-shift.
template <MoranScalar Scalar>
[[nodiscard]] Scalar log_sum_exp(std::span<const Scalar> log_values) {
    if (log_values.empty()) {
        return -std::numeric_limits<Scalar>::infinity();
    }

    const Scalar max_val = *std::ranges::max_element(log_values);

    if (max_val == -std::numeric_limits<Scalar>::infinity()) {
        return -std::numeric_limits<Scalar>::infinity();
    }

    KahanAccumulator<Scalar> acc;
    for (const auto& v : log_values) {
        acc += std::exp(v - max_val);
    }

    return max_val + std::log(acc.sum());
}

/// Fixation probability from log(gamma_i) via log-sum-exp.
template <MoranScalar Scalar>
[[nodiscard]] Scalar fixation_from_log_gammas(std::span<const Scalar> log_gamma) {
    const std::size_t n_minus_1 = log_gamma.size();
    if (n_minus_1 == 0) {
        return Scalar(1);  // Trivial: population size 1
    }

    std::vector<Scalar> log_prods(n_minus_1);
    log_prods[0] = log_gamma[0];
    for (std::size_t k = 1; k < n_minus_1; ++k) {
        log_prods[k] = log_prods[k - 1] + log_gamma[k];
    }

    std::vector<Scalar> all_terms(n_minus_1 + 1);
    all_terms[0] = Scalar(0);
    std::copy(log_prods.begin(), log_prods.end(), all_terms.begin() + 1);

    const auto log_denom = log_sum_exp<Scalar>(std::span<const Scalar>(all_terms));
    return std::exp(-log_denom);
}

/// Stable (1 - r^{-a}) / (1 - r^{-b}). Falls back to a/b when r ~ 1.
template <MoranScalar Scalar>
[[nodiscard]] Scalar stable_fixation_ratio(Scalar r, Scalar a, Scalar b,
                                           Scalar tol = Scalar(1e-12)) {
    const Scalar log_r = std::log(r);
    if (std::abs(log_r) < tol) {
        return a / b;
    }
    const Scalar num = -std::expm1(-a * log_r);
    const Scalar den = -std::expm1(-b * log_r);

    if (!std::isfinite(num) || !std::isfinite(den)) {
        return std::exp((b - a) * log_r);
    }

    return num / den;
}

}  // namespace moran::numeric
