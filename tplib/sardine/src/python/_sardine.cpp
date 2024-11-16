#include <sardine/url.hpp>
#include <sardine/error.hpp>
#include <sardine/mapper.hpp>
#include <sardine/python/mapper.hpp>
#include <sardine/python/cast/url.hpp>

#include <emu/pybind11.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>

#include <cstddef>
#include <fmt/core.h>

namespace py = pybind11;

namespace sardine
{
    void register_context(py::module_ m);
    void register_shm_managed(py::module_ m);
    void register_shm_sync(py::module_ m);
    void register_region_host(py::module_ m);
    // void register_region_device(py::module_m);

    void register_buffer(py::module_ m);

} // namespace sardine

PYBIND11_MODULE(_sardine, m) {
    using namespace sardine;

#if !defined(NDEBUG)
    fmt::print("info: sardine is module compiled in debug mode\n");
#endif

    m.doc() = "pybind11 _sardine plugin";

    register_context(m);

    register_shm_managed(m.def_submodule("managed", "managed submodule"));
    register_shm_sync(m.def_submodule("sync", "sync submodule"));

    auto region = m.def_submodule("region", "region submodule");
    register_region_host(region.def_submodule("host", "host region submodule"));
    // register_shm_region_device(region.def_submodule("device", "device region submodule"));

    register_buffer(m);

    m.def("numpy_ndarray_from_url", [](url url) -> np_ndarray {
        auto view = EMU_UNWRAP_RES_OR_THROW(detail::bytes_from_url(url, emu::dlpack::device_type_t::kDLCPU));

        auto mapping = EMU_UNWRAP_RES_OR_THROW(make_mapping_descriptor(url.params()));

        auto map = EMU_UNWRAP_RES_OR_THROW(mapper< np_ndarray >::from_mapping_descriptor(mapping, emu::capsule()));

        return map.convert(view);
    }, py::arg("url"));

    m.def("url_of_numpy_ndarray", [](np_ndarray obj, bool allow_local) -> url {
        return EMU_UNWRAP_RES_OR_THROW(sardine::url_of(obj, allow_local));
    }, py::arg("obj"), py::arg("allow_local") = false);


}