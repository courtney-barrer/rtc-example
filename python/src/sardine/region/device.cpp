#include <sardine/shm/region/device.hpp>
#include <sardine/shm/region/device/manager.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/format.h>

namespace py = pybind11;

namespace sh = sardine::shm;

void register_shm_region_device(py::module m)
{

    py::class_<sh::region::cuda::device_handle>(m, "device_handle", py::buffer_protocol())
        .def("memory", [](sh::region::cuda::device_handle& h) -> py::object {
            py::module_ cupy(py::module_::import("cupy"));
            py::module_ cuda(cupy.attr("cuda"));

            auto memory = cuda.attr("UnownedMemory")(
                reinterpret_cast<std::uintptr_t>(h.data),
                h.size, nullptr, cuda.attr("Device")(h.device_id)
            );
            return cuda.attr("MemoryPointer")(memory, 0);
        })
        .def("size", [](sh::region::cuda::device_handle& h) -> std::size_t {
            return h.size;
        })
        .def("device", [](sh::region::cuda::device_handle& h) -> py::object {
            py::module_ cupy(py::module_::import("cupy"));
            py::module_ cuda(cupy.attr("cuda"));

            return cuda.attr("Device")(h.device_id);
        })
        // .def_buffer([](sh::region::cuda::device_handle& h) -> py::buffer_info { // not sure if relevant...
        //     return py::buffer_info(
        //         h.get_address(),
        //         sizeof(std::byte),
        //         py::format_descriptor<std::byte>::format(),
        //         1,
        //         { h.get_size() },
        //         { sizeof(std::byte) }
        //     );
        // })
        ;

    m.def("open", [](const std::string& name) -> sh::region::cuda::device_handle {
        return sh::region::cuda::device::manager::instance().open(name);
    }, py::return_value_policy::reference, py::arg("name"));

    m.def("create", [](const std::string& name, std::size_t size, int device_id) -> sh::region::cuda::device_handle {
        return sh::region::cuda::device::manager::instance().create(name, size, ::cuda::device::get(device_id));
    }, py::return_value_policy::reference, py::arg("name"), py::arg("size"), py::arg("device_id"));

    m.add_object("_cleanup", py::capsule([]{
        sh::region::cuda::device::manager::instance().clear();
    }));

}