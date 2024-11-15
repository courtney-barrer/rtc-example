#pragma once

#include <pybind11/numpy.h>

#include <emu/detail/dlpack_types.hpp>

namespace sardine
{
    //TODO: implement a real caster.

    inline uint8_t code_from_np_types(int numpy_type) {
        namespace py = pybind11;

        switch (numpy_type) {
            case py::detail::npy_api::constants::NPY_BOOL_:
                return kDLUInt;

            case py::detail::npy_api::constants::NPY_INT8_:
            case py::detail::npy_api::constants::NPY_INT16_:
            case py::detail::npy_api::constants::NPY_INT32_:
            case py::detail::npy_api::constants::NPY_INT64_:
                return kDLInt;

            case py::detail::npy_api::constants::NPY_UINT8_:
            case py::detail::npy_api::constants::NPY_UINT16_:
            case py::detail::npy_api::constants::NPY_UINT32_:
            case py::detail::npy_api::constants::NPY_UINT64_:
                return kDLUInt;

            case py::detail::npy_api::constants::NPY_FLOAT_:
            case py::detail::npy_api::constants::NPY_DOUBLE_:
            case py::detail::npy_api::constants::NPY_LONGDOUBLE_:
                return kDLFloat;

            default:
                return kDLOpaqueHandle;
        }

    }

    inline int dlpack_type_to_numpy(const emu::dlpack::data_type_ext_t& dtype) {
        namespace py = pybind11;

        switch (dtype.code) {
            case kDLInt:
                switch (dtype.bits) {
                    case 8: return py::detail::npy_api::constants::NPY_INT8_;
                    case 16: return py::detail::npy_api::constants::NPY_INT16_;
                    case 32: return py::detail::npy_api::constants::NPY_INT32_;
                    case 64: return py::detail::npy_api::constants::NPY_INT64_;
                    default: throw std::invalid_argument("Unsupported DLInt bit width");
                }

            case kDLUInt:
                switch (dtype.bits) {
                    case 8: return py::detail::npy_api::constants::NPY_UINT8_;
                    case 16: return py::detail::npy_api::constants::NPY_UINT16_;
                    case 32: return py::detail::npy_api::constants::NPY_UINT32_;
                    case 64: return py::detail::npy_api::constants::NPY_UINT64_;
                    default: throw std::invalid_argument("Unsupported DLUInt bit width");
                }

            case kDLFloat:
                switch (dtype.bits) {
                    case 32: return py::detail::npy_api::constants::NPY_FLOAT_;
                    case 64: return py::detail::npy_api::constants::NPY_DOUBLE_;
                    case 128: return py::detail::npy_api::constants::NPY_LONGDOUBLE_;
                    default: throw std::invalid_argument("Unsupported DLFloat bit width");
                }

            case kDLOpaqueHandle:
                // kDLOpaqueHandle doesn't have a direct equivalent in NumPy. You can return a special value.
                // For example, return NPY_VOID or create your own custom enum value for opaque handles.
                return py::detail::npy_api::constants::NPY_VOID_;

            default:
                throw std::invalid_argument("Unsupported DLPack type code");
        }
    }

} // namespace sardine
