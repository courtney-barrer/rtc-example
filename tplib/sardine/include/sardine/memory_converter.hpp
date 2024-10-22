#pragma once

#include <sardine/type.hpp>
#include <emu/detail/dlpack_types.hpp>

#include <functional>

namespace sardine
{
    using converter_t = std::function<optional<result<span_b>>(bytes_and_device, emu::dlpack::device_type_t)>;
    using revert_converter_t = std::function<optional<span_b>(span_cb)>;

    void register_converter( converter_t converter, revert_converter_t revert_converter );

    result<span_b> convert_bytes( bytes_and_device input, emu::dlpack::device_type_t requested_device_type );
    optional<span_b> revert_convert_bytes( span_cb input );

} // namespace sardine

#define SARDINE_REGISTER_DEVICE_FINDER(NAME, CONVERTER, REVERT_CONVERTER) \
    extern "C" __attribute__ ((constructor)) void NAME() {                \
        ::sardine::register_converter(CONVERTER, REVERT_CONVERTER);       \
    }
