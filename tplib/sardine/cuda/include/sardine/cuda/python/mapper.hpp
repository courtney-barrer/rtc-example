#pragma once
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <sardine/utility.hpp>
#include <sardine/mapper/base.hpp>
#include <sardine/mapper/mapper_base.hpp>
#include <sardine/buffer/base.hpp>
#include <sardine/python/mapper.hpp>

#include <emu/pybind11.hpp>
#include <emu/pybind11/cast/detail/capsule.hpp>

#include <cuda/api.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

#include <span>
#include <type_traits>

namespace sardine
{

    namespace py = pybind11;

    struct cp_ndarray {
        py::object obj;
    };

    inline bool is_strided(const cp_ndarray& arr) {
        // no strides, not strided...
        return not arr.obj.attr("__cuda_array_interface__")["strides"].is_none();
    }

    template<>
    struct mapper< cp_ndarray > : mapper< np_ndarray >
    {
        using base_t = mapper< np_ndarray >;

        using view_type = py::object;
        // using convert_type = np_ndarray;
        // using container_type = np_ndarray;

        using base_t::fake_array;
        using base_t::capsule;

        static result< mapper > from_mapping_descriptor(const interface::mapping_descriptor& f, emu::capsule capsule) {
            //TODO: add a lot of checks!
            auto extents = f.extents();

            std::vector<size_t> shape(extents.begin(), extents.end());

            std::vector<int64_t> strides;
            if (f.is_strided()) {
                auto f_strides = f.strides();
                strides.resize(f_strides.size());
                std::ranges::copy(f_strides, strides.begin());
            }

            np_ndarray fake_array(
                py::dtype(dlpack_type_to_numpy(f.data_type())),
                move(shape),
                move(strides),
                reinterpret_cast<void*>(fake_ptr),
                py::str() // dummy parent class to avoid making a copy and reading the fake_ptr.
            );
            return mapper{std::move(fake_array), emu::capsule()};
        }

        static mapper from(const cp_ndarray& value) {

            py::object array = value.obj;
            const auto rank = array.attr("ndim").cast<size_t>();

            std::vector<size_t> shape(rank), strides(rank);
            {
                auto p_shape = array.attr("shape").cast<py::list>();
                auto p_strides = array.attr("strides").cast<py::list>();

                for (size_t i = 0; i < rank; ++i) {
                    shape[i] = p_shape[i].cast<size_t>();
                    strides[i] = p_strides[i].cast<size_t>();
                }

            }

            np_ndarray fake_array(
                array.attr("dtype").cast<py::dtype>(),
                move(shape), move(strides),
                reinterpret_cast<void*>(fake_ptr),
                py::str() // dummy parent class to avoid making a copy and reading the fake_ptr
            );

            return mapper{std::move(fake_array), emu::capsule(std::move(array))};
        }

        using base_t::size;
        using base_t::offset;
        using base_t::lead_stride;

        mapper close_lead_dim() const {
            return submdspan_mapper(py::make_tuple(0));
        }

        py::object convert(span_b buffer) const {
            using namespace py::literals; // to bring in the `_a` literal

            auto ptr = buffer.data() + offset();

            auto i_ptr = reinterpret_cast<std::uintptr_t>(ptr);

            auto device_id = ::cuda::memory::pointer::wrap(ptr).device().id();

            auto cupy = py::module_::import("cupy");
            auto cuda = cupy.attr("cuda");

            auto memory = cuda.attr("UnownedMemory")(
                /* ptr = */ i_ptr, /* size = */ base_t::required_span_size() * fake_array.itemsize(),
                /* owner = */ emu::pybind11::detail::capsule_to_handle(capsule), /* device_id = */ device_id
            );

            //TODO: get the start of the provided device pointer and then specify the right offset.
            auto memory_ptr = cuda.attr("MemoryPointer")(memory, /* offset = */ 0);

            const auto rank = fake_array.ndim();

            std::vector<ssize_t> shape(rank), strides(rank);
            std::copy_n(fake_array.shape(), rank, shape.begin());
            std::copy_n(fake_array.strides(), rank, strides.begin());

            // There is no way for now to return read only array.
            return cupy.attr("ndarray")(
                "memptr"_a=memory_ptr,
                "dtype"_a=fake_array.dtype(),
                "shape"_a=shape,
                "strides"_a=strides
            );
        }

        static span_b as_bytes(cp_ndarray value) {

            py::object cai = value.obj.attr("__cuda_array_interface__");

            auto data = cai["data"].cast<py::tuple>();

            auto ptr = data[0].cast<uintptr_t>();

            auto size = value.obj.attr("size").cast<size_t>();

            auto itemsize = value.obj.attr("itemsize").cast<size_t>();

            return std::span{reinterpret_cast<std::byte*>(ptr), size * itemsize};

        }


        using base_t::mapping_descriptor;

        mapper submdspan_mapper(py::handle args) const
        {
            py::dict scope;
            scope["mapping"] = fake_array;
            scope["slices"] = args;

            auto result = py::cast<np_ndarray>(py::eval("mapping[slices]", scope));

            return mapper{result, capsule};

        }
    };

namespace buffer
{

    template <typename Derived>
    struct mapper_base< cp_ndarray, Derived > : sardine::mapper< cp_ndarray >
    {
        using mapper_t = sardine::mapper< cp_ndarray >;

        // using element_type = typename V::element_type;

        mapper_base(const mapper_t & a) : mapper_t(a) {}
        mapper_base(const mapper_base &) = default;

        mapper_t       & mapper()       { return *this; }
        mapper_t const & mapper() const { return *this; }

        auto close_lead_dim() const {
            return submdspan(py::make_tuple(0));
        }

        auto submdspan(py::handle args) const {
            return self().clone_with_new_mapper(this->submdspan_mapper(args));
        }

    private:
        const Derived &self() const { return *static_cast<const Derived *>(this); }
    };

} // namespace buffer

} // namespace sardine
