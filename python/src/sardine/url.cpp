#include <pybind11/pytypes.h>
#include <sardine/url.hpp>
#include <sardine/error.hpp>
#include <sardine/mapper.hpp>
#include <sardine/python/mapper.hpp>
#include <sardine/python/cast/url.hpp>

#include <emu/pybind11.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <fmt/core.h>

namespace py = pybind11;

namespace sardine
{

    void register_url(py::module_ m) {

        m.def("from_url", [](url url, py::type type) -> py::object {
            // // np.float32 is not a dtype. Calling `py::dtype::from_args(data_type)` is equivalent
            // // to np.dtype(data_type) and also accept types such as int, float, etc.
            // auto dtype = py::dtype::from_args(data_type);

            // if (py::hasattr(type, "__from_url__"))
            //     return type.attr("__from_url__")(url);

            // TODO write another bytes_from_url specific for sardine python module
            // that takes a dtype as argument
            auto view = EMU_UNWRAP_OR_THROW(bytes_from_url(url, type));

            py::type ndarray_type = py::module_::import("numpy").attr("ndarray");

            if ( type.is(ndarray_type) ) {
                auto mapping = EMU_UNWRAP_OR_THROW(make_mapping_descriptor(url.params()));

                auto map = EMU_UNWRAP_OR_THROW(mapper< py_array >::from_mapping_descriptor(mapping, emu::capsule()));

                return map.convert(view);
            }

            emu::throw_error(error::python_type_not_supported);
        }, py::arg("url"), py::arg("type"));

        m.def("url_of", [](py::object obj, bool allow_local) -> url {
            // if (py::hasattr(obj, "__url_of__"))
            //     return obj.attr("__url_of__")();
            // else
                return EMU_UNWRAP_OR_THROW(sardine::url_of(obj.cast<py::array>(), allow_local));
                // throw std::runtime_error(fmt::format("{} does not have __url_of__", obj.type()));
        }, py::arg("obj"), py::arg("allow_local") = false);

        m.def("url_of", [](py::array obj, bool allow_local) -> url {
            return EMU_UNWRAP_OR_THROW(sardine::url_of(obj, allow_local));
        }, py::arg("obj"), py::arg("allow_local") = false);

    }
} // namespace sardine
