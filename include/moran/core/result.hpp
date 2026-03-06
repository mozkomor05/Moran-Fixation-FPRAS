#pragma once
/// @file result.hpp
/// @brief Error handling via std::expected (C++23).
///
/// All fallible functions return Result<T> = std::expected<T, MoranError>.
/// Python bindings convert MoranError to Python exceptions at the boundary.

#include <cstdint>
#include <expected>
#include <format>
#include <source_location>
#include <string>
#include <string_view>

namespace moran {

enum class ErrorCode : std::uint8_t {
    InvalidPopulationSize,  ///< N <= 0 or N > max supported
    InvalidFitness,         ///< r <= 0 or NaN
    InvalidGraph,           ///< Graph is empty, disconnected, or malformed
    InvalidInitialState,    ///< Initial mutant count out of range
    NumericalInstability,   ///< Detected NaN or Inf in computation
    MaxStepsExceeded,       ///< Simulation exceeded step limit (FPRAS abort)
    InvalidConfig,          ///< Algorithm configuration parameter is out of range
};

/// Structured error with code, message, and source location.
struct MoranError {
    ErrorCode code;
    std::string message;
    std::source_location location;

    MoranError(const ErrorCode c, std::string msg,
               const std::source_location loc = std::source_location::current())
        : code(c), message(std::move(msg)), location(loc) {}

    [[nodiscard]] std::string what() const {
        return std::format("{}: {}", error_code_name(code), message);
    }

    [[nodiscard]] std::string debug_what() const {
        return std::format("[{}:{}] {}: {}", location.file_name(), location.line(),
                           error_code_name(code), message);
    }

    static constexpr std::string_view error_code_name(const ErrorCode c) {
        switch (c) {
            case ErrorCode::InvalidPopulationSize:
                return "InvalidPopulationSize";
            case ErrorCode::InvalidFitness:
                return "InvalidFitness";
            case ErrorCode::InvalidGraph:
                return "InvalidGraph";
            case ErrorCode::InvalidInitialState:
                return "InvalidInitialState";
            case ErrorCode::NumericalInstability:
                return "NumericalInstability";
            case ErrorCode::MaxStepsExceeded:
                return "MaxStepsExceeded";
            case ErrorCode::InvalidConfig:
                return "InvalidConfig";
            default:
                return "Unknown";
        }
    }
};

template <typename T>
using Result = std::expected<T, MoranError>;

inline std::unexpected<MoranError> make_error(
    const ErrorCode code,
    std::string message,  // NOLINT(performance-unnecessary-value-param) -- moved below
    const std::source_location loc = std::source_location::current()) {
    return std::unexpected(MoranError(code, std::move(message), loc));
}

}  // namespace moran
