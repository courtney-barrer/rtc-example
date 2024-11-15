#include <sardine/url.hpp>
#include <sardine/error.hpp>
#include <sardine/mapper.hpp>
#include <sardine/cuda/python/mapper.hpp>
#include <sardine/python/cast/url.hpp>

#include <emu/pybind11.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>

#include <cstddef>
#include <fmt/core.h>


namespace py = pybind11;

PYBIND11_MODULE(_sardinecuda, m) {
    using namespace sardine;

#if !defined(NDEBUG)
    fmt::print("info: sardinecuda is module compiled in debug mode\n");
#endif

    m.doc() = "pybind11 _sardinecuda plugin";

    m.def("cupy_ndarray_from_url", [](url url) -> py::object {
        auto view = EMU_UNWRAP_RES_OR_THROW(detail::bytes_from_url(url, emu::dlpack::device_type_t::kDLCUDA));

        auto mapping = EMU_UNWRAP_RES_OR_THROW(make_mapping_descriptor(url.params()));

        auto map = EMU_UNWRAP_RES_OR_THROW(mapper< cp_ndarray >::from_mapping_descriptor(mapping, emu::capsule()));

        return map.convert(view);
    }, py::arg("url"));

    m.def("url_of_cupy_ndarray", [](py::object obj, bool allow_local) -> url {
        return EMU_UNWRAP_RES_OR_THROW(sardine::url_of(cp_ndarray{obj}, allow_local));
    }, py::arg("obj"), py::arg("allow_local") = false);


}
