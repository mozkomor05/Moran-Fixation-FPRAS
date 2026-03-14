/// @file test_numeric.cpp
/// @brief Tests for core numeric utilities (Kahan summation, log-sum-exp).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include <moran/core/numeric.hpp>

using namespace moran::numeric;

// ---------------------------------------------------------------------------
// Kahan summation tests
// ---------------------------------------------------------------------------

TEST(KahanSummation, ExactForSmallSums) {
    KahanAccumulator<double> acc;
    acc += 1.0;
    acc += 2.0;
    acc += 3.0;
    EXPECT_DOUBLE_EQ(acc.sum(), 6.0);
}

TEST(KahanSummation, BetterThanNaiveForManySmallValues) {
    // Sum 10 million values of 0.1: naive gives large rounding error,
    // Kahan gives accurate result.
    constexpr std::size_t N = 10'000'000;
    KahanAccumulator<double> kahan;
    double naive = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
        kahan += 0.1;
        naive += 0.1;
    }

    const double expected = 1'000'000.0;
    EXPECT_NEAR(kahan.sum(), expected, 1e-6);
    // Naive sum will have larger error (typically ~1e-2)
    EXPECT_GT(std::abs(naive - expected), std::abs(kahan.sum() - expected));
}

TEST(KahanSummation, SpanOverload) {
    std::vector<double> vals = {1e15, 1.0, -1e15, 1.0};
    // Naive sum: 1e15 + 1 - 1e15 + 1 = 0 (catastrophic cancellation)
    // Kahan: 2.0 (correct)
    EXPECT_NEAR(kahan_sum<double>(std::span<const double>(vals)), 2.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Log-sum-exp tests
// ---------------------------------------------------------------------------

TEST(LogSumExp, SimpleCase) {
    std::vector<double> vals = {std::log(1.0), std::numbers::ln2, std::log(3.0)};
    // log(1 + 2 + 3) = log(6)
    EXPECT_NEAR(log_sum_exp<double>(std::span<const double>(vals)),
                std::log(6.0), 1e-14);
}

TEST(LogSumExp, LargeValues) {
    // exp(1000) would overflow, but log-sum-exp should handle it
    std::vector<double> vals = {1000.0, 1000.0 + std::numbers::ln2};
    // log(exp(1000) + exp(1000 + log(2))) = log(exp(1000) * (1 + 2))
    // = 1000 + log(3)
    EXPECT_NEAR(log_sum_exp<double>(std::span<const double>(vals)),
                1000.0 + std::log(3.0), 1e-10);
}

TEST(LogSumExp, NegativeInfinity) {
    std::vector<double> vals = {-std::numeric_limits<double>::infinity()};
    EXPECT_EQ(log_sum_exp<double>(std::span<const double>(vals)),
              -std::numeric_limits<double>::infinity());
}

TEST(LogSumExp, EmptyInput) {
    std::vector<double> vals;
    EXPECT_EQ(log_sum_exp<double>(std::span<const double>(vals)),
              -std::numeric_limits<double>::infinity());
}

// ---------------------------------------------------------------------------
// Stable fixation ratio tests
// ---------------------------------------------------------------------------

TEST(StableFixationRatio, NeutralCase) {
    // r = 1: ratio should approach a/b
    EXPECT_NEAR(stable_fixation_ratio(1.0 + 1e-15, 1.0, 10.0), 0.1, 1e-6);
}

TEST(StableFixationRatio, StandardCase) {
    // r = 2, a = 1, b = 5: (1 - 2^{-1}) / (1 - 2^{-5}) = 0.5 / 0.96875
    const double expected = 0.5 / (1.0 - std::pow(2.0, -5.0));
    EXPECT_NEAR(stable_fixation_ratio(2.0, 1.0, 5.0), expected, 1e-14);
}

// ---------------------------------------------------------------------------
// Fixation from log gammas
// ---------------------------------------------------------------------------

TEST(FixationFromLogGammas, NeutralDrift) {
    // For neutral evolution (r=1), gamma_i = 1 for all i,
    // so log(gamma_i) = 0.
    // phi = 1 / (1 + N-1) = 1/N
    constexpr std::size_t N = 100;
    std::vector<double> log_gamma(N - 1, 0.0);
    const auto phi = fixation_from_log_gammas<double>(
        std::span<const double>(log_gamma));
    EXPECT_NEAR(phi, 1.0 / N, 1e-14);
}

TEST(FixationFromLogGammas, AdvantageousMutant) {
    // r = 2, N = 10: gamma_i = 1/r = 0.5
    // phi = (1 - 1/2) / (1 - 1/2^10) = 0.5 / (1 - 1/1024)
    constexpr std::size_t N = 10;
    constexpr double r = 2.0;
    std::vector<double> log_gamma(N - 1, std::log(1.0 / r));
    const auto phi = fixation_from_log_gammas<double>(
        std::span<const double>(log_gamma));
    const double expected = (1.0 - 1.0 / r) / (1.0 - std::pow(1.0 / r, N));
    EXPECT_NEAR(phi, expected, 1e-12);
}
