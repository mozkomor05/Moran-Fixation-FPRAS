#include <moran/core/result.hpp>
#include <moran/core/types.hpp>

#include <format>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace {
PyObject* MoranError = nullptr;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
PyObject* InvalidInputError =
    nullptr;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
PyObject* MaxStepsExceededError =
    nullptr;                         // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
PyObject* NumericalError = nullptr;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
}  // namespace

void bind_graph(py::module_& m);
void bind_algorithms(py::module_& m);

PYBIND11_MODULE(_moran, m) {
    m.doc() = "libmoran: Moran process fixation probability library.";
    m.attr("__version__") = "0.1.0";

    MoranError = PyErr_NewExceptionWithDoc("_moran.MoranError",
                                           "Base exception for all Moran process library errors.",
                                           PyExc_RuntimeError, nullptr);
    m.attr("MoranError") = py::handle(MoranError);

    InvalidInputError =
        PyErr_NewExceptionWithDoc("_moran.InvalidInputError",
                                  "Raised when input parameters are invalid.", MoranError, nullptr);
    m.attr("InvalidInputError") = py::handle(InvalidInputError);

    MaxStepsExceededError = PyErr_NewExceptionWithDoc(
        "_moran.MaxStepsExceededError", "Raised when a simulation exceeds its step budget.",
        MoranError, nullptr);
    m.attr("MaxStepsExceededError") = py::handle(MaxStepsExceededError);

    NumericalError = PyErr_NewExceptionWithDoc(
        "_moran.NumericalError", "Raised when numerical instability is detected (NaN, Inf).",
        MoranError, nullptr);
    m.attr("NumericalError") = py::handle(NumericalError);

    py::enum_<moran::Method>(m, "Method", "Algorithm used for computation.")
        .value("exact_well_mixed", moran::Method::exact_well_mixed)
        .value("exact_r_equals_1", moran::Method::exact_r_equals_1)
        .value("exact_isothermal_regular", moran::Method::exact_isothermal_regular)
        .value("mc_naive", moran::Method::mc_naive)
        .value("fpras_chatterjee", moran::Method::fpras_chatterjee)
        .value("fpras_goldberg", moran::Method::fpras_goldberg);

    py::enum_<moran::ErrorCode>(m, "ErrorCode")
        .value("InvalidPopulationSize", moran::ErrorCode::InvalidPopulationSize)
        .value("InvalidFitness", moran::ErrorCode::InvalidFitness)
        .value("InvalidGraph", moran::ErrorCode::InvalidGraph)
        .value("InvalidInitialState", moran::ErrorCode::InvalidInitialState)
        .value("NumericalInstability", moran::ErrorCode::NumericalInstability)
        .value("MaxStepsExceeded", moran::ErrorCode::MaxStepsExceeded)
        .value("InvalidConfig", moran::ErrorCode::InvalidConfig);

    py::class_<moran::FixationResult>(m, "FixationResult",
                                      "Result of a fixation probability computation.")
        .def_readonly("estimate", &moran::FixationResult::estimate)
        .def_readonly("ci_lower", &moran::FixationResult::ci_lower)
        .def_readonly("ci_upper", &moran::FixationResult::ci_upper)
        .def_readonly("method", &moran::FixationResult::method)
        .def_readonly("epsilon", &moran::FixationResult::epsilon)
        .def_readonly("delta", &moran::FixationResult::delta)
        .def_readonly("samples", &moran::FixationResult::samples)
        .def_readonly("steps_total", &moran::FixationResult::steps_total)
        .def_readonly("steps_effective", &moran::FixationResult::steps_effective)
        .def_readonly("runs_aborted", &moran::FixationResult::runs_aborted)
        .def_readonly("elapsed_seconds", &moran::FixationResult::elapsed_seconds)
        .def_readonly("seed_used", &moran::FixationResult::seed_used)
        .def("to_dict",
             [](const moran::FixationResult& r) {
                 py::dict d;
                 d["estimate"] = r.estimate;
                 d["ci_lower"] = r.ci_lower;
                 d["ci_upper"] = r.ci_upper;
                 d["method"] = std::string(moran::method_name(r.method));
                 d["epsilon"] = r.epsilon;
                 d["delta"] = r.delta;
                 d["samples"] = r.samples;
                 d["steps_total"] = r.steps_total;
                 d["steps_effective"] = r.steps_effective;
                 d["runs_aborted"] = r.runs_aborted;
                 d["elapsed_seconds"] = r.elapsed_seconds;
                 d["seed_used"] = r.seed_used;
                 return d;
             })
        .def("__repr__",
             [](const moran::FixationResult& r) {
                 return std::format(
                     "FixationResult(estimate={:.6g}, CI=[{:.6g}, {:.6g}], "
                     "method={}, samples={}, elapsed={:.3f}s)",
                     r.estimate, r.ci_lower, r.ci_upper, moran::method_name(r.method), r.samples,
                     r.elapsed_seconds);
             })
        .def("__float__", [](const moran::FixationResult& r) { return r.estimate; });

    py::class_<moran::DegreeStats>(m, "DegreeStats")
        .def_readonly("min_degree", &moran::DegreeStats::min_degree)
        .def_readonly("max_degree", &moran::DegreeStats::max_degree)
        .def_readonly("avg_degree", &moran::DegreeStats::avg_degree)
        .def_readonly("is_regular", &moran::DegreeStats::is_regular)
        .def_readonly("num_edges", &moran::DegreeStats::num_edges)
        .def("__repr__", [](const moran::DegreeStats& s) {
            return std::format("DegreeStats(min={}, max={}, avg={:.2f}, regular={}, edges={})",
                               s.min_degree, s.max_degree, s.avg_degree,
                               s.is_regular ? "True" : "False", s.num_edges);
        });

    bind_graph(m);
    bind_algorithms(m);
}

PyObject* get_moran_error() {
    return MoranError;
}
PyObject* get_invalid_input_error() {
    return InvalidInputError;
}
PyObject* get_max_steps_exceeded_error() {
    return MaxStepsExceededError;
}
PyObject* get_numerical_error() {
    return NumericalError;
}
