#include <sardine/json.hpp>
#include <sardine/python/managed_helper.hpp>
#include <sardine/python/url_helper.hpp>

#include <pybind11/pybind11.h>

#include <iterator>
#include <string_view>

namespace py = pybind11;

namespace sa = sardine;

sa::json::impl_t as_json(py::handle obj, sa::json::allocator alloc);

sa::json::impl_t as_json_list(py::list list, sa::json::allocator alloc)
{
    sa::json::impl_t json(jsoncons::json_array_arg, jsoncons::semantic_tag::none, alloc);
    for (auto item : list)
        json.push_back(as_json(item, alloc));

    return json;
}

sa::json::impl_t as_json_dict(py::dict dict, sa::json::allocator alloc)
{
    sa::json::impl_t json(jsoncons::json_object_arg, jsoncons::semantic_tag::none, alloc);
    for (auto item : dict)
        json.insert_or_assign(item.first.cast<std::string>(), as_json(item.second, alloc));

    return json;
}


sa::json::impl_t as_json(py::handle obj, sa::json::allocator alloc)
{
    if      (py::isinstance<py::str>(obj))
        return sa::json::impl_t(obj.cast<std::string>(), alloc);
    else if (py::isinstance<py::int_>(obj))
        return sa::json::impl_t(obj.cast<int>(), alloc);
    else if (py::isinstance<py::float_>(obj))
        return sa::json::impl_t(obj.cast<double>(), alloc);
    else if (py::isinstance<py::bool_>(obj))
        return sa::json::impl_t(obj.cast<bool>(), alloc);
    else if (py::isinstance<py::dict>(obj))
        return as_json_dict(obj.cast<py::dict>(), alloc);
    else if (py::isinstance<py::list>(obj))
        return as_json_list(obj.cast<py::list>(), alloc);
    else
        throw std::runtime_error("Unknown type");
}

py::object as_pyobject(const sa::json_t& json)
{
    if      (json->is_string())
        return py::cast(json->as_string_view());
    else if (json->is_double())
        return py::cast(json->as_double());
    else if (json->is_number())
        return py::cast(json->as<long long>());
    else if (json->is_bool())
        return py::cast(json->as_bool());
    else if (json->is_null())
        return py::none();
    else if (json->is_array() or json->is_object())
        return py::cast(json);
    else
        throw std::runtime_error("Unknown type");
}

void register_shm_json(py::module m)
{
    py::class_<sa::json_t> json(m, "Json");

    json
        .def("__getitem__", [](sa::json_t& j, std::string_view key) -> sa::json_t {
            return j.at(key);
            // return as_pyobject(obj, j.alloc);
        })
        .def("__getitem__", [](sa::json_t& j, std::size_t index) -> sa::json_t {
            return j.at(index);
            // return as_pyobject(obj, j.alloc);
        })
        .def("__setitem__", [](sa::json_t& j, std::string_view key, const sa::json_t& value) {
            j->insert_or_assign(key, *value);
        })
        .def("__setitem__", [](sa::json_t& j, std::string_view key, py::object value) {
            auto json = as_json(value, j.alloc);
            j->insert_or_assign(key, std::move(json));
        })
        .def("__setitem__", [](sa::json_t& j, std::size_t index, const sa::json_t& value) {
            j->at(index) = *value;
        })
        .def("__setitem__", [](sa::json_t& j, std::size_t index, py::object value) {
            auto json = as_json(value, j.alloc);
            j->at(index) = std::move(json);
        })
        .def("__delitem__", [](sa::json_t& j, std::string_view key) {
            j->erase(key);
        })
        .def("__delitem__", [](sa::json_t& j, std::size_t index) {
            auto it = j->array_range().begin();
            std::advance(it, index);
            j->erase(it);
        })
        .def("append", [](sa::json_t& j, const sa::json_t& value) {
            j->push_back(*value);
        })
        .def("append", [](sa::json_t& j, py::object value) {
            auto json = as_json(value, j.alloc);
            j->push_back(json);
        })
        .def("__len__", [](const sa::json_t& j) {
            return j->size();
        })
        .def("__repr__", [](const sa::json_t& j) {
            return fmt::format("json_shm({})", j);
        })
        .def_property("value",
            [](const sa::json_t& j) {
                return as_pyobject(j);
            },
            [](sa::json_t& j, py::object value) {
                auto json = as_json(value, j.alloc);
                j->swap(json);
            }
        );

    sa::register_managed(json, py::return_value_policy::automatic);
    sa::register_url(json, py::return_value_policy::automatic);

}
