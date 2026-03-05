#pragma once
/// @file csr_graph.hpp
/// @brief Compressed Sparse Row (CSR) graph representation.
///
/// Immutable, cache-friendly graph for Moran process simulations.
/// All arrays aligned to 64 bytes for SIMD.  O(|V| + |E|) memory.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <new>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

#include "../core/concepts.hpp"
#include "../core/types.hpp"
#include "graph_utils.hpp"

namespace moran {
namespace detail {

/// Allocator that aligns all allocations to cache-line boundaries (64 bytes).
/// Used by CSRGraph's internal vectors for SIMD-friendly memory access.
template <typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    explicit AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(const std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        const std::size_t bytes = n * sizeof(T);
        const std::size_t alloc_size = (bytes + Alignment - 1) & ~(Alignment - 1);
        return static_cast<T*>(::operator new(alloc_size, std::align_val_t{Alignment}));
    }

    void deallocate(T* p, std::size_t) noexcept {
        // Unsized aligned-delete: allocate() rounds up, so passing the
        // original size to sized-delete would be UB.
        ::operator delete(p, std::align_val_t{Alignment});
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }
};

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

}  // namespace detail

/// Immutable undirected graph in Compressed Sparse Row format.
template <MoranScalar WeightType = double>
class CSRGraph {
public:
    using weight_type = WeightType;

    CSRGraph() = default;

    struct WeightedEdge {
        VertexId src;
        VertexId dst;
        WeightType weight;
    };

    /// Construct from weighted edge list. Each edge appears once; reverse is added automatically.
    explicit CSRGraph(const std::size_t num_vertices, std::span<const WeightedEdge> edges)
        : num_vertices_(num_vertices) {
        build_from_edges(num_vertices, edges);
        compute_stats();
    }

    struct Edge {
        VertexId src;
        VertexId dst;
    };

    /// Construct from unweighted edge list (all weights = 1).
    explicit CSRGraph(const std::size_t num_vertices, std::span<const Edge> edges)
        : num_vertices_(num_vertices) {
        std::vector<WeightedEdge> wedges;
        wedges.reserve(edges.size());
        for (const auto& e : edges) {
            wedges.push_back({e.src, e.dst, WeightType(1)});
        }
        build_from_edges(num_vertices, std::span<const WeightedEdge>(wedges));
        compute_stats();
    }

    [[nodiscard]] std::size_t num_vertices() const noexcept { return num_vertices_; }

    /// Number of undirected edges (each stored twice in CSR).
    [[nodiscard]] std::size_t num_edges() const noexcept { return col_indices_.size() / 2; }

    [[nodiscard]] std::size_t degree(const VertexId v) const noexcept {
        assert(v < num_vertices_);
        return row_offsets_[v + 1] - row_offsets_[v];
    }

    /// Contiguous span over neighbor IDs of v (cache-friendly).
    [[nodiscard]] std::span<const VertexId> neighbors(const VertexId v) const noexcept {
        assert(v < num_vertices_);
        const auto begin = row_offsets_[v];
        const auto end = row_offsets_[v + 1];
        return {col_indices_.data() + begin, static_cast<std::size_t>(end - begin)};
    }

    [[nodiscard]] EdgeIdx row_begin(const VertexId v) const noexcept { return row_offsets_[v]; }
    [[nodiscard]] EdgeIdx row_end(const VertexId v) const noexcept { return row_offsets_[v + 1]; }
    [[nodiscard]] const VertexId* col_indices_data() const noexcept { return col_indices_.data(); }
    [[nodiscard]] const WeightType* edge_weights_data() const noexcept {
        return edge_weights_.data();
    }
    [[nodiscard]] const EdgeIdx* row_offsets_data() const noexcept { return row_offsets_.data(); }

    [[nodiscard]] WeightType edge_weight(const VertexId v, const std::size_t k) const noexcept {
        assert(k < degree(v));
        return edge_weights_[row_offsets_[v] + k];
    }

    /// Sum of edge weights incident to v.
    [[nodiscard]] WeightType weighted_degree(const VertexId v) const noexcept {
        const auto begin = row_offsets_[v];
        const auto end = row_offsets_[v + 1];
        WeightType sum{0};
        for (auto i = begin; i < end; ++i) {
            sum += edge_weights_[i];
        }
        return sum;
    }

    [[nodiscard]] const DegreeStats& degree_stats() const noexcept { return stats_; }

private:
    void compute_stats() {
        if (num_vertices_ > 0) {
            stats_ = compute_degree_stats(*this);
        }
    }

    void build_from_edges(const std::size_t nv, std::span<const WeightedEdge> edges) {
        for (const auto& e : edges) {
            if (e.src >= static_cast<VertexId>(nv) || e.dst >= static_cast<VertexId>(nv)) {
                throw std::invalid_argument("Edge vertex ID out of range for graph");
            }
            if (e.src == e.dst) {
                throw std::invalid_argument("Self-loops are not supported");
            }
        }

        // Prefix-sum to build row_offsets from degree counts
        row_offsets_.resize(nv + 1, 0);
        for (const auto& e : edges) {
            ++row_offsets_[e.src + 1];
            ++row_offsets_[e.dst + 1];
        }
        for (std::size_t i = 1; i <= nv; ++i) {
            row_offsets_[i] += row_offsets_[i - 1];
        }

        const auto total_entries = row_offsets_[nv];
        col_indices_.resize(total_entries);
        edge_weights_.resize(total_entries);

        // Fill CSR arrays: each undirected edge is stored in both directions
        std::vector<EdgeIdx> cursor(row_offsets_.begin(), row_offsets_.end());
        for (const auto& e : edges) {
            col_indices_[cursor[e.src]] = e.dst;
            edge_weights_[cursor[e.src]] = e.weight;
            ++cursor[e.src];

            col_indices_[cursor[e.dst]] = e.src;
            edge_weights_[cursor[e.dst]] = e.weight;
            ++cursor[e.dst];
        }

        // Sort neighbors by vertex ID for deterministic iteration
        for (VertexId v = 0; v < static_cast<VertexId>(nv); ++v) {
            const auto begin = row_offsets_[v];
            const auto end = row_offsets_[v + 1];
            const auto len = end - begin;
            if (len <= 1) {
                continue;
            }

            std::vector<EdgeIdx> idx(len);
            std::ranges::iota(idx, EdgeIdx{0});
            std::ranges::sort(idx, [&](const EdgeIdx a, const EdgeIdx b) {
                return col_indices_[begin + a] < col_indices_[begin + b];
            });

            std::vector<VertexId> sorted_cols(len);
            std::vector<WeightType> sorted_weights(len);
            for (EdgeIdx k = 0; k < len; ++k) {
                sorted_cols[k] = col_indices_[begin + idx[k]];
                sorted_weights[k] = edge_weights_[begin + idx[k]];
            }
            std::ranges::copy(sorted_cols,
                              col_indices_.begin() + static_cast<std::ptrdiff_t>(begin));
            std::ranges::copy(sorted_weights,
                              edge_weights_.begin() + static_cast<std::ptrdiff_t>(begin));
        }
    }

    std::size_t num_vertices_ = 0;
    detail::AlignedVector<EdgeIdx> row_offsets_;
    detail::AlignedVector<VertexId> col_indices_;
    detail::AlignedVector<WeightType> edge_weights_;
    DegreeStats stats_{};
};

}  // namespace moran
