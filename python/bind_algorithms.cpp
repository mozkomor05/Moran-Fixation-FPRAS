#include <moran/moran.hpp>

#include <format>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "helpers.hpp"

namespace py = pybind11;
namespace gs = moran::graph_structured;

using Graph = moran::CSRGraph<double>;

namespace {

moran::SimulationConfig make_config(const double epsilon, const double delta,
                                    const std::uint64_t seed, const int num_threads) {
    if (epsilon <= 0.0 || epsilon >= 1.0) {
        throw py::value_error(std::format("epsilon must be in (0, 1), got {}", epsilon));
    }
    if (delta <= 0.0 || delta >= 1.0) {
        throw py::value_error(std::format("delta must be in (0, 1), got {}", delta));
    }
    if (num_threads < 0) {
        throw py::value_error(std::format("num_threads must be >= 0, got {}", num_threads));
    }
    return {
        .accuracy = {.epsilon = epsilon, .delta = delta},
        .seed = seed,
        .num_threads = num_threads,
    };
}

}  // namespace

void bind_algorithms(py::module_& m) {
    auto alg = m.def_submodule("algorithms", "Fixation probability algorithms");

    // Exact formulas
    auto exact = alg.def_submodule("exact", "Exact fixation probability formulas");

    exact.def("well_mixed", &moran::exact::well_mixed, py::arg("n"), py::arg("r"),
              "Exact well-mixed fixation: (1-1/r)/(1-1/r^n).");

    exact.def("isothermal_regular", &moran::exact::isothermal_regular, py::arg("n"), py::arg("r"),
              "Exact isothermal regular graph fixation (same as well-mixed).");

    exact.def("r_equals_1", &moran::exact::r_equals_1, py::arg("n"),
              "Neutral drift fixation probability: 1/n.");

    exact.def(
        "try_exact",
        [](const Graph& g, const double r) -> py::object {
            auto result = moran::exact::try_exact(g, r);
            if (!result) {
                return py::none();
            }
            return py::cast(*result);
        },
        py::arg("graph"), py::arg("r"), "Try exact formula; returns FixationResult or None.");

    // Well-mixed exact (Result<T> API)
    auto wm = alg.def_submodule("wellmixed", "Well-mixed population formulas");

    wm.def(
        "fixation_probability",
        [](const std::size_t N, const double r) -> double {
            const py::gil_scoped_release release;
            return unwrap_or_throw(moran::exact::fixation_exact<double>(N, r));
        },
        py::arg("N"), py::arg("r"));

    wm.def(
        "fixation_probability_from",
        [](const std::size_t N, const std::size_t i0, const double r) -> double {
            const py::gil_scoped_release release;
            return unwrap_or_throw(moran::exact::fixation_from<double>(N, i0, r));
        },
        py::arg("N"), py::arg("i0"), py::arg("r"));

    // Naive MC
    auto graph_alg =
        alg.def_submodule("graph_structured", "Graph-structured Moran process algorithms");

    graph_alg.def(
        "naive_mc_fixation",
        [](const Graph& graph, const double r, const double epsilon, const double delta,
           const std::uint64_t seed, const int num_threads) {
            auto config = make_config(epsilon, delta, seed, num_threads);
            const py::gil_scoped_release release;
            return unwrap_or_throw(gs::naive_mc_fixation(graph, r, config));
        },
        py::arg("graph"), py::arg("r"), py::arg("epsilon") = 0.1, py::arg("delta") = 0.25,
        py::arg("seed") = 0, py::arg("num_threads") = 0,
        "Naive MC fixation (Diaz et al. 2014, Algorithmica, Theorem 13).");

    // Derived parameters
    auto params = alg.def_submodule("params", "FPRAS parameter derivation");

    py::class_<moran::fpras::DerivedParams>(params, "DerivedParams")
        .def_readonly("samples", &moran::fpras::DerivedParams::samples)
        .def_readonly("per_run_step_limit", &moran::fpras::DerivedParams::per_run_step_limit)
        .def("__repr__", [](const moran::fpras::DerivedParams& p) {
            return std::format("DerivedParams(samples={}, per_run_step_limit={})", p.samples,
                               p.per_run_step_limit);
        });

    params.def(
        "diaz_naive",
        [](const std::size_t n, const double r, const double epsilon, const double delta) {
            return moran::fpras::diaz_naive(n, r, {.epsilon = epsilon, .delta = delta});
        },
        py::arg("n"), py::arg("r"), py::arg("epsilon") = 0.1, py::arg("delta") = 0.25,
        "Compute Diaz naive MC parameters from epsilon/delta.");

    params.def("multiplicative_ci", &moran::fpras::multiplicative_ci, py::arg("estimate"),
               py::arg("epsilon"), "Compute multiplicative CI: [est/(1+eps), est/(1-eps)].");
}
