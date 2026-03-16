#include <array>
#include <gtest/gtest.h>
#include <cmath>

#include <moran/moran.hpp>

using namespace moran;

TEST(NaiveSingle, TerminatesOnSmallGraph) {
    auto g = make_complete_graph<double>(5);
    Xoshiro256StarStar rng(42);
    auto result = graph_structured::simulate_naive_single(g, 1.5, 0, rng, 100'000);
    EXPECT_GT(result.total_steps, 0ULL);
    EXPECT_GE(result.total_steps, result.effective_steps);
}

TEST(NaiveMC, CompleteGraphMatchesWellMixed) {
    constexpr std::size_t N = 10;
    constexpr double r = 1.5;
    auto exact_val = exact::well_mixed(N, r);

    const SimulationConfig config{
        .accuracy = {.epsilon = 0.05, .delta = 0.25},
        .seed = 42, .num_threads = 1};

    auto g = make_complete_graph<double>(N);
    auto result = graph_structured::naive_mc_fixation(g, r, config);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->estimate, exact_val, 0.05);
    EXPECT_EQ(result->method, Method::mc_naive);
    EXPECT_GT(result->samples, 0ULL);
}

TEST(NaiveMC, CycleIsothermal) {
    constexpr std::size_t N = 10;
    constexpr double r = 1.5;
    auto exact_val = exact::well_mixed(N, r);

    const SimulationConfig config{
        .accuracy = {.epsilon = 0.05, .delta = 0.25},
        .seed = 42, .num_threads = 1};

    auto g = make_cycle_graph<double>(N);
    auto result = graph_structured::naive_mc_fixation(g, r, config);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->estimate, exact_val, 0.05);
}

TEST(NaiveMC, NeutralDrift) {
    constexpr std::size_t N = 10;

    const SimulationConfig config{
        .accuracy = {.epsilon = 0.05, .delta = 0.25},
        .seed = 123, .num_threads = 1};

    auto g = make_complete_graph<double>(N);
    auto result = graph_structured::naive_mc_fixation(g, 1.0, config);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->estimate, 1.0 / N, 0.05);
}

TEST(NaiveMC, RejectsInvalidInputs) {
    auto g = make_complete_graph<double>(5);
    const SimulationConfig config{};

    EXPECT_FALSE(graph_structured::naive_mc_fixation(g, 0.0, config).has_value());
    EXPECT_FALSE(graph_structured::naive_mc_fixation(g, -1.0, config).has_value());

    const std::array edges = {CSRGraph<double>::Edge{.src = 0, .dst = 1}, CSRGraph<double>::Edge{.src = 2, .dst = 3}};
    const CSRGraph<double> disc(4, std::span<const CSRGraph<double>::Edge>(edges));
    EXPECT_FALSE(graph_structured::naive_mc_fixation(disc, 1.5, config).has_value());
}

TEST(StarGraph, AmplifiesSelection) {
    constexpr std::size_t N = 15;
    constexpr double r = 2.0;
    const auto well_mixed_val = exact::well_mixed(N, r);

    const SimulationConfig config{
        .accuracy = {.epsilon = 0.1, .delta = 0.25},
        .seed = 42, .num_threads = 1};

    auto star = make_star_graph<double>(N);
    auto result = graph_structured::naive_mc_fixation(star, r, config);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->estimate, well_mixed_val);
    EXPECT_LT(result->estimate, 1.0);
}

TEST(Multithreaded, ConsistentResults) {
    constexpr std::size_t N = 10;
    constexpr double r = 1.5;
    auto exact_val = exact::well_mixed(N, r);
    auto g = make_complete_graph<double>(N);

    const SimulationConfig config_1t{
        .accuracy = {.epsilon = 0.05, .delta = 0.25},
        .seed = 42, .num_threads = 1};

    const SimulationConfig config_mt{
        .accuracy = {.epsilon = 0.05, .delta = 0.25},
        .seed = 42, .num_threads = 4};

    auto result_1t = graph_structured::naive_mc_fixation(g, r, config_1t);
    auto result_mt = graph_structured::naive_mc_fixation(g, r, config_mt);
    ASSERT_TRUE(result_1t.has_value());
    ASSERT_TRUE(result_mt.has_value());

    EXPECT_NEAR(result_1t->estimate, exact_val, 0.05);
    EXPECT_NEAR(result_mt->estimate, exact_val, 0.05);
    EXPECT_EQ(result_1t->samples, result_mt->samples);
}

TEST(StepsTotal, NaiveMCPopulated) {
    auto g = make_complete_graph<double>(8);
    const SimulationConfig config{
        .accuracy = {.epsilon = 0.1, .delta = 0.25},
        .seed = 42, .num_threads = 1};
    auto result = graph_structured::naive_mc_fixation(g, 1.5, config);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->steps_total, 0ULL);
    EXPECT_GT(result->steps_effective, 0ULL);
    EXPECT_GE(result->steps_total, result->steps_effective);
}
