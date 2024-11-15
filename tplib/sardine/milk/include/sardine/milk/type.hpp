#pragma once

extern "C"
{

#include <CommandLineInterface/CLIcore.h>
#include <CommandLineInterface/function_parameters.h>
#include <ImageStreamIO/ImageStruct.h>

} // extern "C"

#include <emu/cstring_view.hpp>
#include <emu/numeric_type.hpp>
#include <emu/span.hpp>

#include <sardine/type.hpp>
#include <sardine/type/url.hpp>

#include <string_view>
#include <cstddef>

namespace sardine::milk
{
    // Utility types
    using std::size_t, std::byte, std::dynamic_extent, std::uint64_t;

    // Span
    using emu::span, sardine::span_b, sardine::span_cb;

    // String support
    using std::string, std::string_view;
    using emu::cstring_view;

    using dtype_t = emu::dlpack::data_type_ext_t;

    namespace dtype = emu::dlpack::dtype;

    using image_int_type_t = int;

    using fps_int_type_t = int;



    // using sardine::create_only;
    // using sardine::open_only;
    // using sardine::open_or_create;

    // enum class fps_type_t : fps_int_type_t {
    //     i32 = FPTYPE_INT32,
    //     u32 = FPTYPE_UINT32,
    //     i64 = FPTYPE_INT64,
    //     u64 = FPTYPE_UINT64,
    //     f32 = FPTYPE_FLOAT32,
    //     f64 = FPTYPE_FLOAT64,
    //     str = FPTYPE_STRING,
    //     bool = FPTYPE_ONOFF
    // };


} // namespace sardine::milk
