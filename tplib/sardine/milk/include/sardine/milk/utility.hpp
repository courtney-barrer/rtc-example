#pragma once

#include <sardine/milk/type.hpp>
#include <sardine/utility.hpp>
#include <emu/assert.hpp>

namespace sardine::milk
{

    constexpr inline result<image_int_type_t> to_image_type(dtype_t type) {
        if (type == dtype::uint8)      return _DATATYPE_UINT8;
        if (type == dtype::int8)       return _DATATYPE_INT8;
        if (type == dtype::uint16)     return _DATATYPE_UINT16;
        if (type == dtype::int16)      return _DATATYPE_INT16;
        if (type == dtype::uint32)     return _DATATYPE_UINT32;
        if (type == dtype::int32)      return _DATATYPE_INT32;
        if (type == dtype::uint64)     return _DATATYPE_UINT64;
        if (type == dtype::int64)      return _DATATYPE_INT64;
        if (type == dtype::float16)    return _DATATYPE_HALF;
        if (type == dtype::float32)    return _DATATYPE_FLOAT;
        if (type == dtype::float64)    return _DATATYPE_DOUBLE;
        if (type == dtype::complex64)  return _DATATYPE_COMPLEX_FLOAT;
        if (type == dtype::complex128) return _DATATYPE_COMPLEX_DOUBLE;

        EMU_COLD_LOGGER("Data type not handled: {}", type);
        return make_unexpected(errc::array_data_type_not_handled);

    }

    constexpr inline dtype_t to_dtype(image_int_type_t type) {
        switch (type)
        {
            case _DATATYPE_UINT8         : return dtype::uint8;
            case _DATATYPE_INT8          : return dtype::int8;
            case _DATATYPE_UINT16        : return dtype::uint16;
            case _DATATYPE_INT16         : return dtype::int16;
            case _DATATYPE_UINT32        : return dtype::uint32;
            case _DATATYPE_INT32         : return dtype::int32;
            case _DATATYPE_UINT64        : return dtype::uint64;
            case _DATATYPE_INT64         : return dtype::int64;
            case _DATATYPE_HALF          : return dtype::float16;
            case _DATATYPE_FLOAT         : return dtype::float32;
            case _DATATYPE_DOUBLE        : return dtype::float64;
            case _DATATYPE_COMPLEX_FLOAT : return dtype::complex64;
            case _DATATYPE_COMPLEX_DOUBLE: return dtype::complex128;
        }
        EMU_UNREACHABLE;
    }

} // namespace sardine::milk
