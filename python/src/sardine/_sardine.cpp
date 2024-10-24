#include <pybind11/pybind11.h>

#include <sardine/python/cast/url.hpp>
// #include <sardine/utility.hpp>
// #include <sardine/value.hpp>

#include <emu/pybind11.hpp>

#include <cstddef>
#include <fmt/core.h>

namespace py = pybind11;

namespace sardine
{
    void register_context(py::module_ m);
    void register_url(py::module_ m);
    void register_shm_managed(py::module_ m);
    // void register_shm_json(py::module_ m);
    void register_shm_semaphore(py::module_ m);
    void register_region_host(py::module_ m);
    // void register_region_device(py::module_m);

    void register_buffer(py::module_ m);

} // namespace sardine


PYBIND11_MODULE(_sardine, m) {
    using namespace sardine;

#if !defined(NDEBUG)
    fmt::print("info: sardine is module compiled in debug mode\n");
#endif

    m.doc() = "nanobind _sardine plugin";

    m.def("fun", [](){
        fmt::print("fun!!\n");
    }, "A function!!");


    register_context(m);
    register_url(m);

    // register_shm_managed(m.def_submodule("managed", "managed submodule"));
    // register_shm_json(m.def_submodule("json", "json submodule"));
    register_shm_semaphore(m.def_submodule("sync", "Synchronization submodule"));

    auto region = m.def_submodule("region", "region submodule");
    register_region_host(region.def_submodule("host", "host region submodule"));
    // register_shm_region_device(region.def_submodule("device", "device region submodule"));

    register_buffer(m);
}
