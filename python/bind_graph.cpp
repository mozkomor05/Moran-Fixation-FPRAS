#include <moran/graph/csr_graph.hpp>
#include <moran/graph/graph_factories.hpp>
#include <moran/graph/graph_utils.hpp>
#include <moran/graph/graph_validation.hpp>

#include <format>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using Graph = moran::CSRGraph<double>;

namespace {

void check_vertex_bounds(const Graph& g, moran::VertexId v) {
    if (v >= static_cast<moran::VertexId>(g.num_vertices())) {
        throw py::index_error(
            std::format("vertex {} out of range for graph with {} vertices", v, g.num_vertices()));
    }
}

}  // namespace

void bind_graph(py::module_& m) {
    auto gm = m.def_submodule("graph", "Graph types and construction");

    py::class_<Graph>(gm, "CSRGraph", "Compressed Sparse Row graph for population structure.")
        .def(py::init(
                 [](const std::size_t n,
                    const py::array_t<moran::VertexId, py::array::forcecast | py::array::c_style>&
                        src,
                    const py::array_t<moran::VertexId, py::array::forcecast | py::array::c_style>&
                        dst) {
                     const auto s = src.unchecked<1>();
                     const auto d = dst.unchecked<1>();
                     if (s.shape(0) != d.shape(0)) {
                         throw py::value_error("src and dst arrays must have same length");
                     }
                     std::vector<Graph::Edge> edges(static_cast<std::size_t>(s.shape(0)));
                     for (py::ssize_t i = 0; i < s.shape(0); ++i) {
                         edges[i] = {.src = s(i), .dst = d(i)};
                     }
                     try {
                         return Graph(n, std::span<const typename Graph::Edge>(edges));
                     } catch (const std::invalid_argument& e) {
                         throw py::value_error(e.what());
                     }
                 }),
             py::arg("num_vertices"), py::arg("src"), py::arg("dst"))
        .def("num_vertices", &Graph::num_vertices)
        .def("num_edges", &Graph::num_edges)
        .def(
            "degree",
            [](const Graph& g, moran::VertexId v) {
                check_vertex_bounds(g, v);
                return g.degree(v);
            },
            py::arg("v"), "Return the degree of vertex v.")
        .def(
            "neighbors",
            [](const Graph& g, moran::VertexId v) {
                check_vertex_bounds(g, v);
                auto nbrs = g.neighbors(v);
                return std::vector<moran::VertexId>(nbrs.begin(), nbrs.end());
            },
            py::arg("v"), "Return neighbor vertex IDs as a list.")
        .def("is_connected", [](const Graph& g) { return moran::is_connected(g); })
        .def("degree_stats", &Graph::degree_stats)
        .def("__repr__",
             [](const Graph& g) {
                 return std::format("CSRGraph(vertices={}, edges={})", g.num_vertices(),
                                    g.num_edges());
             })
        .def("__len__", &Graph::num_vertices)
        .def(
            "__contains__",
            [](const Graph& g, const py::int_& v_obj) {
                auto v = v_obj.cast<std::int64_t>();
                return v >= 0 && static_cast<std::uint64_t>(v) < g.num_vertices();
            },
            py::arg("v"))
        .def("__iter__",
             [](const Graph& g) {
                 return py::iter(py::module_::import("builtins").attr("range")(g.num_vertices()));
             })

        .def_static(
            "from_networkx",
            [](const py::object& nx_graph) -> Graph {
                if (py::cast<bool>(nx_graph.attr("is_directed")())) {
                    throw py::value_error("CSRGraph only supports undirected graphs");
                }
                if (py::cast<bool>(nx_graph.attr("is_multigraph")())) {
                    throw py::value_error("CSRGraph does not support multigraphs");
                }
                const auto n = static_cast<std::size_t>(py::len(nx_graph.attr("nodes")));
                const py::module_ nx = py::module_::import("networkx");
                const py::object relabeled = nx.attr("convert_node_labels_to_integers")(nx_graph);
                std::vector<Graph::Edge> edge_list;
                for (const auto& e : relabeled.attr("edges")()) {
                    const auto tup = py::cast<py::tuple>(e);
                    edge_list.push_back({.src = tup[0].cast<moran::VertexId>(),
                                         .dst = tup[1].cast<moran::VertexId>()});
                }
                return Graph(n, std::span<const typename Graph::Edge>(edge_list));
            },
            py::arg("nx_graph"), "Construct from a NetworkX undirected graph.")

        .def_static(
            "from_scipy_sparse",
            [](const py::object& matrix) -> Graph {
                const py::module_ sp = py::module_::import("scipy.sparse");
                const py::object csr = sp.attr("csr_array")(matrix);
                const auto shape = csr.attr("shape").cast<py::tuple>();
                const auto rows = shape[0].cast<std::size_t>();
                const auto cols = shape[1].cast<std::size_t>();
                if (rows != cols) {
                    throw py::value_error(
                        std::format("Adjacency matrix must be square, got {}x{}", rows, cols));
                }
                const py::object diff = csr - csr.attr("T");
                if (diff.attr("nnz").cast<std::size_t>() != 0) {
                    throw py::value_error("Adjacency matrix must be symmetric");
                }
                const auto n = rows;
                auto ip =
                    csr.attr("indptr")
                        .cast<py::array_t<int64_t, py::array::forcecast | py::array::c_style>>()
                        .unchecked<1>();
                auto idx =
                    csr.attr("indices")
                        .cast<py::array_t<int64_t, py::array::forcecast | py::array::c_style>>()
                        .unchecked<1>();
                std::vector<Graph::Edge> edges;
                for (std::size_t i = 0; i < n; ++i) {
                    for (auto j = ip(static_cast<py::ssize_t>(i));
                         j < ip(static_cast<py::ssize_t>(i + 1)); ++j) {
                        const auto col = static_cast<std::size_t>(idx(j));
                        if (col == i) {
                            throw py::value_error(
                                std::format("Self-loops not allowed (vertex {})", i));
                        }
                        if (col > i) {
                            edges.push_back({.src = static_cast<moran::VertexId>(i),
                                             .dst = static_cast<moran::VertexId>(col)});
                        }
                    }
                }
                return Graph(n, std::span<const typename Graph::Edge>(edges));
            },
            py::arg("matrix"), "Construct from a scipy sparse adjacency matrix.")

        .def(
            "to_networkx",
            [](const Graph& g) {
                const py::module_ nx = py::module_::import("networkx");
                py::object G = nx.attr("Graph")();
                G.attr("add_nodes_from")(
                    py::module_::import("builtins").attr("range")(g.num_vertices()));
                const auto n = g.num_vertices();
                const auto* offsets = g.row_offsets_data();
                const auto* indices = g.col_indices_data();
                py::list edge_list;
                for (std::size_t u = 0; u < n; ++u) {
                    for (auto pos = offsets[u]; pos < offsets[u + 1]; ++pos) {
                        if (auto v = indices[pos]; v > u) {
                            edge_list.append(py::make_tuple(u, static_cast<std::size_t>(v)));
                        }
                    }
                }
                G.attr("add_edges_from")(edge_list);
                return G;
            },
            "Convert to a NetworkX graph.")

        .def(
            "to_scipy_sparse",
            [](const Graph& g) {
                const py::module_ sp = py::module_::import("scipy.sparse");
                const auto n = static_cast<py::ssize_t>(g.num_vertices());
                const auto nnz = static_cast<py::ssize_t>(g.row_offsets_data()[g.num_vertices()]);
                py::array_t<std::int64_t> indptr(n + 1);
                auto ip = indptr.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i <= n; ++i) {
                    ip(i) = static_cast<std::int64_t>(g.row_offsets_data()[i]);
                }
                py::array_t<std::int64_t> indices(nnz);
                auto idx = indices.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < nnz; ++i) {
                    idx(i) = static_cast<std::int64_t>(g.col_indices_data()[i]);
                }
                py::array_t<double> data(nnz);
                auto d = data.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < nnz; ++i) {
                    d(i) = static_cast<double>(g.edge_weights_data()[i]);
                }
                return sp.attr("csr_array")(py::make_tuple(data, indices, indptr),
                                            py::arg("shape") = py::make_tuple(n, n));
            },
            "Convert to a SciPy CSR sparse adjacency matrix.");

    gm.def("complete_graph", &moran::make_complete_graph<double>, py::arg("n"),
           "Create complete graph K_n.");
    gm.def("cycle_graph", &moran::make_cycle_graph<double>, py::arg("n"),
           "Create cycle graph C_n.");
    gm.def("star_graph", &moran::make_star_graph<double>, py::arg("n"),
           "Create star graph S_n (vertex 0 is hub).");
    gm.def("double_star_graph", &moran::make_double_star_graph<double>, py::arg("a"), py::arg("b"),
           "Create double star graph D(a,b): two hubs with a and b leaves.");
}
