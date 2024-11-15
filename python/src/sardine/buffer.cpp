#include <sardine/buffer.hpp>
#include <sardine/python/cast/url.hpp>
#include <sardine/python/mapper.hpp>

namespace py = pybind11;

namespace sardine
{

    // using np_array = py::ndarray<py::numpy, py::device::cpu>;

    using np_view_t = view_t<py_array>;
    using np_producer = producer<py_array, host_context>;
    using np_consumer = consumer<py_array, host_context>;
    using np_box = box<py_array, host_context>;

    void register_buffer(py::module_ m) {

        py::class_<sardine::np_view_t>(m, "View")
            .def(py::init<sardine::py_array&>())
            .def("__getitem__", &sardine::np_view_t::submdspan)
            .def("view", &sardine::np_view_t::view) // TODO: returns a reference. check if still correct
            .def("url", &sardine::np_view_t::url);

        py::class_<sardine::np_producer>(m, "Producer")
            .def(py::init<sardine::py_array&>())
            .def("__getitem__", &sardine::np_producer::submdspan)
            .def("view", &sardine::np_producer::view) // TODO: returns a reference. check if still correct
            .def("url", &sardine::np_producer::url)
            .def("send", &sardine::np_producer::send);

        py::class_<sardine::np_consumer>(m, "Consumer")
            .def(py::init<sardine::py_array&>())
            .def("__getitem__", &sardine::np_consumer::submdspan)
            .def("view", &sardine::np_consumer::view) // TODO: returns a reference. check if still correct
            .def("url", &sardine::np_consumer::url)
            .def("recv", &sardine::np_consumer::recv);

        // py::class_<sardine::np_box>(m, "box")
        //     .def(py::init<sardine::py_array>());

        // m.def("make_producer", &sardine::make_producer<host_context, py_array>);
        // m.def("make_consumer", &sardine::make_consumer<host_context, py_array>);
        // m.def("make_box", &sardine::make_box<host_context, py_array>);
    }
} // namespace sardine
