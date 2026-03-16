#include <gtest/gtest.h>
#include <cmath>

#include <moran/exact/formulas.hpp>

using namespace moran::exact;

TEST(FixationProbability, NeutralDrift) {
    for (const std::size_t N : {2, 5, 10, 50, 100, 1000}) {
        const auto result = fixation_exact<double>(N, 1.0);
        ASSERT_TRUE(result.has_value()) << result.error().what();
        EXPECT_NEAR(*result, 1.0 / static_cast<double>(N), 1e-12)
            << "Failed for N=" << N;
    }
}

TEST(FixationProbability, AdvantagedMutant) {
    const auto result = fixation_exact<double>(10, 2.0);
    ASSERT_TRUE(result.has_value());
    const double expected = (1.0 - 0.5) / (1.0 - std::pow(0.5, 10));
    EXPECT_NEAR(*result, expected, 1e-12);
}

TEST(FixationProbability, DisadvantagedMutant) {
    const auto result = fixation_exact<double>(10, 0.5);
    ASSERT_TRUE(result.has_value());
    const double expected = (1.0 - 2.0) / (1.0 - std::pow(2.0, 10));
    EXPECT_NEAR(*result, expected, 1e-12);
}

TEST(FixationProbability, SingleIndividual) {
    const auto result = fixation_exact<double>(1, 2.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(*result, 1.0);
}

TEST(FixationProbability, InvalidInputs) {
    EXPECT_FALSE(fixation_exact<double>(0, 1.0).has_value());
    EXPECT_FALSE(fixation_exact<double>(10, 0.0).has_value());
    EXPECT_FALSE(fixation_exact<double>(10, -1.0).has_value());
}

TEST(FixationGeneral, ConstantFitnessMatchesExact) {
    constexpr std::size_t N = 15;
    constexpr double r = 1.3;

    const auto exact = fixation_exact<double>(N, r);
    const auto general = fixation_general<double>(N, 1,
        [r](std::size_t) { return 1.0 / r; });

    ASSERT_TRUE(exact.has_value());
    ASSERT_TRUE(general.has_value());
    EXPECT_NEAR(*general, *exact, 1e-10);
}

TEST(FixationProbability, RejectsNaNFitness) {
    EXPECT_FALSE(fixation_exact<double>(10, std::numeric_limits<double>::quiet_NaN()).has_value());
}

TEST(FixationProbability, RejectsInfFitness) {
    EXPECT_FALSE(fixation_exact<double>(10, std::numeric_limits<double>::infinity()).has_value());
}

TEST(FixationFrom, BoundaryConditions) {
    const auto r0 = fixation_from<double>(10, 0, 1.5);
    ASSERT_TRUE(r0.has_value());
    EXPECT_DOUBLE_EQ(*r0, 0.0);

    const auto rN = fixation_from<double>(10, 10, 1.5);
    ASSERT_TRUE(rN.has_value());
    EXPECT_DOUBLE_EQ(*rN, 1.0);

    const auto r_too_big = fixation_from<double>(10, 11, 1.5);
    EXPECT_FALSE(r_too_big.has_value());
}
