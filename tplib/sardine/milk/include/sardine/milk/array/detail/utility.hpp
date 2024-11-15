#ifndef CACAO_DETAIL_UTILITY_H
#define CACAO_DETAIL_UTILITY_H

#include <cacao/detail/type.hpp>

#include <emu/assert.hpp>
#include <emu/string.hpp>
#include <emu/optional.hpp>

#include <ImageStruct.h>

namespace cacao
{

    constexpr inline image_int_type_t image_type(type_t type) {
        switch (type)
        {
            case type_t:: u8: return _DATATYPE_UINT8;
            case type_t:: i8: return _DATATYPE_INT8;
            case type_t::u16: return _DATATYPE_UINT16;
            case type_t::i16: return _DATATYPE_INT16;
            case type_t::u32: return _DATATYPE_UINT32;
            case type_t::i32: return _DATATYPE_INT32;
            case type_t::u64: return _DATATYPE_UINT64;
            case type_t::i64: return _DATATYPE_INT64;
            case type_t::f16: return _DATATYPE_HALF;
            case type_t::f32: return _DATATYPE_FLOAT;
            case type_t::f64: return _DATATYPE_DOUBLE;
            case type_t::c16: EMU_UNREACHABLE;
            case type_t::c32: return _DATATYPE_COMPLEX_FLOAT;
            case type_t::c64: return _DATATYPE_COMPLEX_DOUBLE;
        }
        EMU_UNREACHABLE;
    }

    constexpr inline type_t octopus_type(image_int_type_t type) {
        switch (type)
        {
            case _DATATYPE_UINT8         : return type_t:: u8;
            case _DATATYPE_INT8          : return type_t:: i8;
            case _DATATYPE_UINT16        : return type_t::u16;
            case _DATATYPE_INT16         : return type_t::i16;
            case _DATATYPE_UINT32        : return type_t::u32;
            case _DATATYPE_INT32         : return type_t::i32;
            case _DATATYPE_UINT64        : return type_t::u64;
            case _DATATYPE_INT64         : return type_t::i64;
            case _DATATYPE_HALF          : return type_t::f16;
            case _DATATYPE_FLOAT         : return type_t::f32;
            case _DATATYPE_DOUBLE        : return type_t::f64;
            case _DATATYPE_COMPLEX_FLOAT : return type_t::c32;
            case _DATATYPE_COMPLEX_DOUBLE: return type_t::c64;
        }
        EMU_UNREACHABLE;
    }

} // namespace cacao

#endif //CACAO_DETAIL_UTILITY_H
