#ifndef FPS_DETAIL_TYPE_H
#define FPS_DETAIL_TYPE_H

#include <octopus/detail/type.h>

extern "C" {
    #include <CommandLineInterface/CLIcore.h>
    #include <CommandLineInterface/CLIcore/CLIcore_datainit.h>
    #include <CommandLineInterface/processtools.h>
    #include <CommandLineInterface/function_parameters.h>
    #include <CommandLineInterface/fps/fps_FPSremove.h>
    #include <CommandLineInterface/fps/fps_GetParamIndex.h>
}

#include <string>
#include <variant>

namespace fps
{

    using octopus::i32;
    using octopus::i64;
    using octopus::f32;
    using octopus::f64;

    using fps_int_type_t = int;

    enum class type_t : fps_int_type_t {
        i32 = FPTYPE_INT32,
        i64 = FPTYPE_INT64,
        f32 = FPTYPE_FLOAT32,
        f64 = FPTYPE_FLOAT64,
        str = FPTYPE_STRING
    };

    using key_t = const std::string &;

    using var_t = std::variant<std::monostate, i32, i64, f32, f64, std::string>;

} // namespace fps

#endif //FPS_DETAIL_TYPE_H