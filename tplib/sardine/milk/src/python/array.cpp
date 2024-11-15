#include <sardine/milk/array.hpp>
#include <sardine/python/dtype.hpp>
#include <sardine/python/cast/url.hpp>

#include <emu/pybind11/cast/span.hpp>
#include <emu/pybind11/cast/cstring_view.hpp>
#include <emu/pybind11/cast/detail/capsule.hpp>

#include <pybind11/stl.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace sardine::milk
{

    enum class Region {
        base,
        current,
        next
    };

    py::object get_array_view(const ArrayBase& array, Region region)
    {
        span_b bytes;
        std::vector<ssize_t> shape;
        auto extents = array.extents();

        switch (region) {
            case Region::base:
                bytes = array.base_bytes();
                shape = {
                    static_cast<ssize_t>(array.buffer_nb()),
                    static_cast<ssize_t>(extents[0]), static_cast<ssize_t>(extents[1]) //TODO: it maube not be the right order.
                };
                break;
            case Region::current:
                bytes = array.current_bytes();
                shape = {static_cast<ssize_t>(extents[0]), static_cast<ssize_t>(extents[1])};
                break;
            case Region::next:
                bytes = array.next_bytes();
                shape = {static_cast<ssize_t>(extents[0]), static_cast<ssize_t>(extents[1])};
                break;
        }
        auto dt = py::dtype(dlpack_type_to_numpy(array.dtype()));

        auto capsule_handle = emu::pybind11::detail::capsule_to_handle(emu::capsule(array.handle_ptr()));

#ifdef SARDINE_CUDA
        if (array.location() >= 0) {
            using namespace py::literals; // to bring in the `_a` literal

            auto cupy = py::module_::import("cupy");
            auto cuda = cupy.attr("cuda");

            auto i_ptr = reinterpret_cast<std::uintptr_t>(bytes.data());

            auto memory = cuda.attr("UnownedMemory")(
                /* ptr = */ i_ptr, /* size = */ bytes.size(),
                /* owner = */ capsule_handle, /* device_id = */ array.location()
            );

            auto memory_ptr = cuda.attr("MemoryPointer")(memory, /* offset = */ 0);

            return cupy.attr("ndarray")(
                "memptr"_a=memory_ptr,
                "dtype"_a=dt,
                "shape"_a=shape,
                "strides"_a= py::none()
            );
        }
#endif
        return py::array(
            dt, shape,
            reinterpret_cast<const void*>(bytes.data()),
            capsule_handle
        );
    }

    void register_array(py::module_ m)
    {
        //TODO: returns data as memoryview instead.
        py::class_<ArrayBase> array(m, "Array");

        array
            .def("wait", &ArrayBase::wait)
            .def("notify", &ArrayBase::notify)
            .def("next", &ArrayBase::next)
            .def("sync_ptr", &ArrayBase::sync_ptr)
            .def_property_readonly("available", &ArrayBase::available)
            .def_property_readonly("current_index", &ArrayBase::current_index)
            .def_property_readonly("next_index", &ArrayBase::next_index)
            .def_property_readonly("buffer_nb", &ArrayBase::buffer_nb)
            .def_property_readonly("rank", &ArrayBase::rank)
            .def_property_readonly("size", &ArrayBase::size)
            .def_property_readonly("shape", &ArrayBase::extents)
            .def_property_readonly("dtype", [](const ArrayBase& array){
                return py::dtype(dlpack_type_to_numpy(array.dtype()));
            })
            .def_property_readonly("location", &ArrayBase::location)
            .def_property_readonly("name", &ArrayBase::name)
            .def("__url_of__", &ArrayBase::url)
            .def_static("__from_url__", [](url u) { return from_url(u); }, py::arg("url"))
            .def_property_readonly("base_view", [](const ArrayBase& array){
                return get_array_view(array, Region::base);
            })
            .def_property_readonly("current_view", [](const ArrayBase& array){
                return get_array_view(array, Region::current);
            })
            .def_property_readonly("next_view", [](const ArrayBase& array){
                return get_array_view(array, Region::next);
            })
            .def("__repr__", [](const ArrayBase& array){
                return fmt::format("<Array name='{}' shape={} dtype={} location={} buffer_nb={} @{}->{}>",
                    array.name(), array.extents(), array.dtype(), array.location(), array.buffer_nb(), array.current_index(), array.distance());
            })
        ;


        m.def("open", [](string name) -> ArrayBase {
            return open(name);
        }, py::arg("name"));

        m.def("create", [](string name, std::vector<size_t> extents, size_t buffer_nb, py::object data_type, int location, bool overwrite) -> ArrayBase {
            auto dtype = py::dtype::from_args(data_type);
            auto type = emu::dlpack::data_type_ext_t{
                .code = code_from_np_types(dtype.num()),
                .bits = static_cast<uint64_t>(dtype.itemsize()) * CHAR_BIT,
                .lanes = 1
            };
            return create(name, extents, buffer_nb, type, location, overwrite);
        }, py::arg("name"), py::arg("extents"), py::arg("buffer_nb"), py::arg("dtype"), py::arg("location") = -1, py::arg("overwrite") = true);

        m.def("open_or_create", [](string name, std::vector<size_t> extents, size_t buffer_nb, py::object data_type, int location, bool overwrite) -> ArrayBase {
            auto dtype = py::dtype::from_args(data_type);
            auto type = emu::dlpack::data_type_ext_t{
                .code = code_from_np_types(dtype.num()),
                .bits = static_cast<uint64_t>(dtype.itemsize()) * CHAR_BIT,
                .lanes = 1
            };
            return open_or_create(name, extents, buffer_nb, type, location, overwrite);
        }, py::arg("name"), py::arg("extents"), py::arg("buffer_nb"), py::arg("dtype"), py::arg("location") = -1, py::arg("overwrite") = true);


    }

} // namespace sardine::milk
