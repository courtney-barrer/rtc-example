#include <sardine/config.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <boost/json.hpp>
#include <emu/pybind11.hpp>

namespace py = pybind11;
namespace json = EMU_BOOST_NAMESPACE::json;

namespace sardine::json
{

    value as_json(py::handle obj);

    array as_json_list(py::list list)
    {
        array json;
        for (auto item : list)
            json.push_back(as_json(item));

        return json;
    }

    object as_json_dict(py::dict dict)
    {
        object json;
        for (auto item : dict)
            json.emplace(item.first.cast<std::string>(), as_json(item.second));

        return json;
    }

    value as_json(py::handle obj)
    {
        if      (py::isinstance<py::str>(obj))
            return value(obj.cast<std::string>());

        else if (py::isinstance<py::int_>(obj))
            return value(obj.cast<long long>());

        else if (py::isinstance<py::float_>(obj))
            return value(obj.cast<double>());

        else if (py::isinstance<py::bool_>(obj))
            return value(obj.cast<bool>());

        else if (py::isinstance<py::dict>(obj))
            return as_json_dict(obj.cast<py::dict>());

        else if (py::isinstance<py::list>(obj))
            return as_json_list(obj.cast<py::list>());

        else
            throw std::runtime_error("Unknown type");
    }

    py::object as_pyobject(const value& json);

    py::object as_pyobject(const object& json)
    {
        py::dict res;
        for  (const auto& pair : json)
            res[py::str(pair.key())] = as_pyobject(pair.value());
        return res;
    }

    py::object as_pyobject(const array& json)
    {
        py::list res;
        for (const auto& item : json)
            res.append(as_pyobject(item));
        return res;
    }


    py::object as_pyobject(const value& json)
    {
        if      (json.is_string())
            return py::cast(std::string_view(json.as_string()));
        else if (json.is_double())
            return py::cast(json.as_double());
        else if (json.is_int64())
            return py::cast(json.as_int64());
        else if (json.is_uint64())
            return py::cast(json.as_uint64());
        else if (json.is_bool())
            return py::cast(json.as_bool());
        else if (json.is_null())
            return py::none();
        else if (json.is_array())
            return as_pyobject(json.as_array());
        else if (json.is_object())
            return as_pyobject(json.as_object());
        else
            throw std::runtime_error("Unknown type");
    }


} // namespace sardine::json

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

namespace detail
{

template <>
struct type_caster<sardine::json::value> {
    // Conversion part 1 (Python -> C++)
    bool load(handle src, bool) {
        value = sardine::json::as_json(src);

        return true;
    }

    // Conversion part 2 (C++ -> Python)
    static handle cast(sardine::json::value src, return_value_policy, handle) {
        return sardine::json::as_pyobject(src);
    }

    PYBIND11_TYPE_CASTER(sardine::json::value, _("boost::json::value"));
};

template <>
struct type_caster<sardine::json::object> {
    // Conversion part 1 (Python -> C++)
    bool load(handle src, bool) {

        value = sardine::json::as_json_dict(src.cast<py::dict>());

        return true;
    }

    // Conversion part 2 (C++ -> Python)
    static handle cast(sardine::json::object src, return_value_policy, handle) {
        return sardine::json::as_pyobject(src);
    }

    PYBIND11_TYPE_CASTER(sardine::json::object, _("boost::json::object"));
};


template <>
struct type_caster<sardine::json::array> {
    // Conversion part 1 (Python -> C++)
    bool load(handle src, bool) {
        value = sardine::json::as_json_list(src.cast<py::list>());

        return true;
    }

    // Conversion part 2 (C++ -> Python)
    static handle cast(sardine::json::array src, return_value_policy, handle) {
        return sardine::json::as_pyobject(src);
    }

    PYBIND11_TYPE_CASTER(sardine::json::array, _("boost::json::array"));
};


} // namespace detail

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
