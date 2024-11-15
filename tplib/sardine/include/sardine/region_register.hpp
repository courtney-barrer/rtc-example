#pragma once

#include <sardine/type.hpp>
#include <sardine/type/url.hpp>
#include <emu/detail/dlpack_types.hpp>

#include <functional>
#include <string>

namespace sardine::registry
{
    // A function that converts a region of bytes to a url.
    // Since there is no way to know which function can actually convert the bytes to a url,
    // optional nullopt indicates that the function did not convert the bytes to a url.
    // The result error state indicates that it was the correct function but it failed.
    using bytes_to_url_t = std::function<optional<result<url>>(span_cb)>;

    // A function that converts a url to a region of bytes.
    using url_to_bytes_t = std::function<result<bytes_and_device>(url_view)>;

    void register_url_region_converter( std::string scheme_name, bytes_to_url_t btu, url_to_bytes_t utb );

    optional<result<url>> url_from_bytes( span_cb data );
    result<bytes_and_device> url_to_bytes( url_view u );

} // namespace sardine::registry

#ifndef NDEBUG // Debug mode

#include <fmt/format.h>

#define SARDINE_REGISTER_URL_CONVERTER(NAME, SCHEME, BTU, UTB)      \
    extern "C" __attribute__ ((constructor)) void NAME() {  \
        fmt::print("Registering url converter: " #NAME "\n"); \
        ::sardine::registry::register_url_region_converter(SCHEME, BTU, UTB); \
    }

#else // Release mode

#define SARDINE_REGISTER_URL_CONVERTER(NAME, SCHEME, BTU, UTB)      \
    extern "C" __attribute__ ((constructor)) void NAME() {  \
        ::sardine::registry::register_url_region_converter(SCHEME, BTU, UTB); \
    }

#endif
