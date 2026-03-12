#pragma once

#include <moran/core/result.hpp>

#include <pybind11/pybind11.h>

// Forward declarations for custom exception types (defined in pymoran.cpp).
PyObject* get_moran_error();
PyObject* get_invalid_input_error();
PyObject* get_max_steps_exceeded_error();
PyObject* get_numerical_error();

[[noreturn]] inline void throw_moran_error(const moran::MoranError& err) {
    const pybind11::gil_scoped_acquire acquire;
    const auto msg = err.what();  // includes ErrorCode + message
    PyObject* exc = nullptr;
    switch (err.code) {
        case moran::ErrorCode::MaxStepsExceeded:
            exc = get_max_steps_exceeded_error();
            break;
        case moran::ErrorCode::NumericalInstability:
            exc = get_numerical_error();
            break;
        default:
            exc = get_invalid_input_error();
            break;
    }
    PyErr_SetString(exc, msg.c_str());
    throw pybind11::error_already_set();
}

template <typename T>
T unwrap_or_throw(moran::Result<T>&& result) {
    if (result) {
        return std::move(*result);
    }
    throw_moran_error(result.error());
}

/// Specialization for Result<void> -- just check for error.
inline void unwrap_or_throw(moran::Result<void>&& result) {
    if (!result) {
        throw_moran_error(result.error());
    }
}
