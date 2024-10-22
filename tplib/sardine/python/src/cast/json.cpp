// #include <sardine/type.hpp>

// #include <pybind11/pybind11.h>

// namespace py = pybind11;

// void register_json_shm(py::module& m)
// {
//     py::class_<sardine::ipc::JsonManager>(m, "JsonManager")
//         .def("create", &sardine::ipc::JsonManager::create)
//         .def("open_or_create", &sardine::ipc::JsonManager::open_or_create)
//         .def("at", &sardine::ipc::JsonManager::at);

//     m.add_object("json_manager", py::cast(&sardine::ipc::JsonManager::instance()));
// }