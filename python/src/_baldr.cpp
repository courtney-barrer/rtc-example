#include <pybind11/pybind11.h>

#include <sardine/python/url_helper.hpp>

#include <baldr/type.hpp>
#include <baldr/baldr.hpp>
#include <sardine/cache.hpp>
#include <sardine/python/cast/json.hpp>
#include <sardine/python/cast/url.hpp>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(_baldr, m) {
    fmt::print("pybind11 _baldr plugin!!!!\n");

    m.doc() = "pybind11 _baldr plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");

    using namespace baldr;

    py::class_<node::Camera>(m, "Camera")
        .def("__call__", [](node::Camera& self){ self(); })
        .def_static("init", &init_camera)
    ;

    py::enum_<cmd>(m, "Cmd")
        .value("pause", cmd::pause)
        .value("run", cmd::run)
        .value("exit", cmd::exit)
        .export_values();

    py::class_<Command> command(m, "Command");

    command
    .def_static("create", [](cmd init_value) -> Command& {
        return sardine::cache::request<Command>(init_value);
    }, py::arg("init_value") = cmd::pause, py::return_value_policy::reference)
    .def("send", [](Command& command, cmd new_cmd){
        command = new_cmd;
        command.notify_all();
    })
    .def("recv", [](Command& command){
        return command.load();
    });

    sardine::register_url(command);
    

}
