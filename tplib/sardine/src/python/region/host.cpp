#include <emu/assert.hpp>
#include <sardine/region/host.hpp>
#include <sardine/region/host/manager.hpp>
#include <sardine/python/mapper.hpp>

#include <emu/pybind11/cast/span.hpp>
#include <emu/pybind11/cast/cstring_view.hpp>

#include <pybind11/stl.h>

// #include <pybind11/stl/string.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>

namespace py = pybind11;

namespace sardine
{

    void register_region_host(py::module_ m)
    {
        //TODO: returns data as memoryview instead.

        m.def("open", [](std::string name) -> std::span<uint8_t> {
            return emu::as_t<uint8_t>(region::host::open(name));
        }, py::arg("name"));

        m.def("create", [](std::string name, std::size_t size) -> std::span<uint8_t> {
            return emu::as_t<uint8_t>(region::host::create(name, size));
        }, py::arg("name"), py::arg("size"));

        m.def("open_or_create", [](std::string name, std::size_t size) -> std::span<uint8_t> {
            return emu::as_t<uint8_t>(region::host::open_or_create(name, size));
        }, py::arg("name"), py::arg("size"));


        m.def("open", [](std::string name, py::object data_type) -> np_ndarray {
            auto dtype = py::dtype::from_args(data_type);
            auto bytes = region::host::open(name);

            return np_ndarray(
                /* dtype = */ dtype,
                /* shape = */ std::vector<py::ssize_t>(1, bytes.size() / dtype.itemsize() ),
                /* ptr = */ v_ptr_of(bytes),
                py::str() // dummy handle to avoid copying the data.
            );
        }, py::arg("name"), py::arg("dtype"));

        m.def("create", [](std::string name, std::vector<py::ssize_t> shape, py::object data_type) -> np_ndarray {
            auto dtype = py::dtype::from_args(data_type);
            size_t bytes_size = dtype.itemsize(); for (auto e : shape) bytes_size *= e;

            auto bytes = region::host::create(name, bytes_size);

            return np_ndarray(
                /* dtype = */ dtype,
                /* shape = */ move(shape),
                /* ptr = */ v_ptr_of(bytes),
                py::str() // dummy handle to avoid copying the data.
            );
        }, py::arg("name"), py::arg("shape"), py::arg("dtype"));

        m.def("open_or_create", [](std::string name, std::vector<py::ssize_t> shape, py::object data_type) -> np_ndarray {
            auto dtype = py::dtype::from_args(data_type);
            size_t bytes_size = dtype.itemsize(); for (auto e : shape) bytes_size *= e;

            auto bytes = region::host::open_or_create(name, bytes_size);

            // In case of opening an exiting host shm.
            EMU_TRUE_OR_THROW(bytes_size == bytes.size(), error::host_incompatible_shape);

            return np_ndarray(
                /* dtype = */ dtype,
                /* shape = */ move(shape),
                /* ptr = */ v_ptr_of(bytes),
                py::str() // dummy handle to avoid copying the data.
            );
        }, py::arg("name"), py::arg("shape"), py::arg("dtype"));

    }

} // namespace sardine
