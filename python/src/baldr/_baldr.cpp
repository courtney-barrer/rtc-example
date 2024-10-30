#include <baldr/camera.hpp>
#include <baldr/rtc.hpp>
#include <baldr/dm.hpp>

#include <sardine/python/url_helper.hpp>

#include <baldr/type.hpp>
#include <baldr/baldr.hpp>
#include <sardine/cache.hpp>
#include <sardine/python/cast/json.hpp>
#include <sardine/python/cast/url.hpp>

#include <emu/pybind11/cast/span.hpp>

#include <pybind11/pybind11.h>



namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(_baldr, m) {
    m.doc() = "pybind11 _baldr plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");

    using namespace baldr;

    py::class_<node::Camera>(m, "Camera")
        .def("__call__", &node::Camera::LOOK_last_frame)
        .def_static("init", &init_camera)
    ;

    py::class_<node::RTC>(m, "RTC")
        .def("__call__", [](node::RTC& self){ self(); })
        .def_static("init", &init_rtc)
    ;

    py::class_<node::DM>(m, "DM")
        .def("__call__", [](node::DM& self){ self(); })
        .def_static("init", &init_dm)
    ;

    py::enum_<cmd>(m, "Cmd")
        .value("pause", cmd::pause)
        .value("run", cmd::run)
        .value("step", cmd::step)
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
        })
        .def("run", [](Command& command){
            command = cmd::run;
            command.notify_all();
        })
        .def("step", [](Command& command){
            command = cmd::step;
            command.notify_all();
        })
        .def("pause", [](Command& command){
            command = cmd::pause;
            command.notify_all();
        })
        .def("exit", [](Command& command){
            command = cmd::exit;
            command.notify_all();
        });

    sardine::register_url(command);

    py::enum_<LockState>(m, "LockState")
        .value("Locked", LockState::Locked)
        .value("Unlocked", LockState::Unlocked)
        .export_values();

    py::class_<SpinLock> spin_lock(m, "SpinLock");

    spin_lock
        .def_static("create", [](LockState init_value) -> SpinLock& {
            return sardine::cache::request<SpinLock>(init_value);
        }, py::arg("init_value") = LockState::Locked, py::return_value_policy::reference)
        .def("lock", &SpinLock::lock)
        .def("try_lock", &SpinLock::try_lock)
        .def("unlock", &SpinLock::unlock)
        .def_property_readonly("value", [](const SpinLock& sl) -> LockState {
            return sl.state.load();
        });

    sardine::register_url(spin_lock);


}
