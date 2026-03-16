
#include <gtest/gtest.h>
#include <moran/graph/csr_graph.hpp>
#include <moran/graph/graph_factories.hpp>
#include <moran/graph/graph_utils.hpp>
#include <moran/graph/graph_validation.hpp>

using namespace moran;

// CSR Graph construction

TEST(CSRGraph, CompleteGraph) {
    auto g = make_complete_graph<double>(5);
    EXPECT_EQ(g.num_vertices(), 5);
    EXPECT_EQ(g.num_edges(), 10);  // C(5,2) = 10
    for (VertexId v = 0; v < 5; ++v) {
        EXPECT_EQ(g.degree(v), 4);
    }
}

TEST(CSRGraph, CycleGraph) {
    auto g = make_cycle_graph<double>(6);
    EXPECT_EQ(g.num_vertices(), 6);
    EXPECT_EQ(g.num_edges(), 6);
    for (VertexId v = 0; v < 6; ++v) {
        EXPECT_EQ(g.degree(v), 2);
    }
}

TEST(CSRGraph, StarGraph) {
    auto g = make_star_graph<double>(5);
    EXPECT_EQ(g.num_vertices(), 5);
    EXPECT_EQ(g.num_edges(), 4);
    EXPECT_EQ(g.degree(0), 4);  // Hub
    for (VertexId v = 1; v < 5; ++v) {
        EXPECT_EQ(g.degree(v), 1);  // Leaves
    }
}

TEST(CSRGraph, NeighborsSorted) {
    auto g = make_complete_graph<double>(4);
    for (VertexId v = 0; v < 4; ++v) {
        auto nbrs = g.neighbors(v);
        for (std::size_t i = 1; i < nbrs.size(); ++i) {
            EXPECT_LT(nbrs[i - 1], nbrs[i]);
        }
    }
}

// Graph utilities

TEST(GraphUtils, ConnectedGraph) {
    auto g = make_complete_graph<double>(5);
    EXPECT_TRUE(is_connected(g));
}

TEST(GraphUtils, DisconnectedGraph) {
    // Build two disconnected components
    CSRGraph<double>::Edge edges[] = {{0, 1}, {2, 3}};
    const CSRGraph<double> g(4, std::span<const CSRGraph<double>::Edge>(edges, 2));
    EXPECT_FALSE(is_connected(g));
}

TEST(GraphUtils, DegreeStats) {
    auto g = make_star_graph<double>(5);
    auto stats = g.degree_stats();
    EXPECT_EQ(stats.min_degree, 1);
    EXPECT_EQ(stats.max_degree, 4);
    EXPECT_FALSE(stats.is_regular);
    EXPECT_EQ(stats.num_edges, 4);
}

TEST(GraphUtils, RegularGraph) {
    auto g = make_complete_graph<double>(4);
    auto stats = g.degree_stats();
    EXPECT_TRUE(stats.is_regular);
}

TEST(GraphUtils, IsothermalComplete) {
    auto g = make_complete_graph<double>(5);
    EXPECT_TRUE(is_isothermal(g));
}
