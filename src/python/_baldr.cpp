#include <baldr/type.hpp>
#include <baldr/baldr.hpp>
#include <baldr/utility/runner.hpp>

#include <baldr/utility/command.hpp>
#include <baldr/utility/component_info.hpp>

#include <sardine/region/managed.hpp>
#include <sardine/python/url_helper.hpp>
#include <sardine/python/managed_helper.hpp>

#include <sardine/cache.hpp>
#include <sardine/python/cast/json.hpp>
#include <sardine/python/cast/url.hpp>

#include <emu/pybind11/cast/span.hpp>
#include <emu/pybind11/cast/cstring_view.hpp>

#include <pybind11/pybind11.h>



namespace py = pybind11;


PYBIND11_MODULE(_baldr, m) {
    m.doc() = "pybind11 _baldr plugin"; // optional module docstring

    using namespace baldr;

    py::enum_<Command>(m, "Command")
        .value("pause", Command::pause)
        .value("run", Command::run)
        .value("step", Command::step)
        .value("exit", Command::exit)
        .export_values();

    py::enum_<Status>(m, "Status")
        .value("none", Status::none)
        .value("acquired", Status::acquired)
        .value("running", Status::running)
        .value("pausing", Status::pausing)
        .value("exited", Status::exited)
        .value("crashed", Status::crashed)
        .export_values();

    // py::enum_<cmd>(m, "Cmd")
    //     .value("pause", cmd::pause)
    //     .value("run", cmd::run)
    //     .value("step", cmd::step)
    //     .value("exit", cmd::exit)
    //     .export_values();


    // py::class_<node::Camera>(m, "Camera")
    //     .def("__call__", &node::Camera::get_last_frame)
    //     .def_static("init", &init_camera)
    // ;

    py::class_<node::RTC>(m, "RTC")
        .def("__call__", [](node::RTC& self){ self(); })
        .def_static("init", &init_rtc)
    ;

    py::class_<node::DM>(m, "DM")
        .def("__call__", [](node::DM& self){ self(); })
        .def_static("init", &init_dm)
    ;

    // py::class_<CommandAtom> command(m, "CommandAtom");

    // command
    //     .def_static("create", [](cmd init_value) -> CommandAtom& {
    //         return sardine::cache::request<CommandAtom>(init_value);
    //     }, py::arg("init_value") = cmd::pause, py::return_value_policy::reference)
    //     .def("send", [](CommandAtom& command, cmd new_cmd){
    //         command = new_cmd;
    //         command.notify_all();
    //     })
    //     .def("recv", [](CommandAtom& command){
    //         return command.load();
    //     })
    //     .def("run", [](CommandAtom& command){
    //         command = cmd::run;
    //         command.notify_all();
    //     })
    //     .def("step", [](CommandAtom& command){
    //         command = cmd::step;
    //         command.notify_all();
    //     })
    //     .def("pause", [](CommandAtom& command){
    //         command = cmd::pause;
    //         command.notify_all();
    //     })
    //     .def("exit", [](CommandAtom& command){
    //         command = cmd::exit;
    //         command.notify_all();
    //     });

    // sardine::register_url(command);

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
        .def("value", [](const SpinLock& sl, size_t idx) -> LockState {
            return sl.state[idx].load();
        });

    sardine::register_url(spin_lock);

    py::class_<ComponentInfo> ci(m, "ComponentInfo");

    ci
        .def("acquire", &ComponentInfo::acquire)
        .def("release", &ComponentInfo::release)
        .def_property_readonly("name", &ComponentInfo::name)
        .def_readonly("pid", &ComponentInfo::pid)
        .def_property("status", &ComponentInfo::status, &ComponentInfo::set_status)
        .def_property("cmd", &ComponentInfo::cmd, &ComponentInfo::set_cmd)
        .def_property_readonly("loop_count", [](const ComponentInfo& ci) -> size_t {
            return ci.loop_count.load();
        })
        .def("run", [](ComponentInfo& ci){ ci.set_cmd(Command::run); })
        .def("step", [](ComponentInfo& ci){ ci.set_cmd(Command::step); })
        .def("pause", [](ComponentInfo& ci){ ci.set_cmd(Command::pause); })
        .def("exit", [](ComponentInfo& ci){ ci.set_cmd(Command::exit); })
    ;

    sardine::register_url(ci);

    py::class_<ComponentInfoManager>(m, "ComponentInfoManager")
        .def_static("get", [] () -> ComponentInfoManager& {
            auto managed = sardine::region::managed::open_or_create("baldr_component_info", 1024*1024);

            return managed.open_or_create<ComponentInfoManager>("baldr_component_info");
        }, py::return_value_policy::reference)
        .def("create_component_info", &ComponentInfoManager::create_component_info,
            py::return_value_policy::reference)
        .def_property_readonly("components", [](ComponentInfoManager& cim) -> py::dict {
            sardine::sync::scoped_lock lock(cim.mut);
            py::dict d;
            for (auto& ci : cim.components) {
                d[py::str(std::string_view(ci.name()))] = py::cast(ci, py::return_value_policy::reference);
            }
            return d;
        })
        .def("clear", &ComponentInfoManager::clear)
    ;



}
