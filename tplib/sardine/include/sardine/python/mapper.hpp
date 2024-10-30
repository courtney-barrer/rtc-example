#pragma once

#include <dlpack/dlpack.h>
#include <sardine/utility.hpp>
#include <sardine/mapper/base.hpp>
#include <sardine/mapper/mapper_base.hpp>
#include <sardine/buffer/base.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eval.h>

#include <span>
#include <type_traits>

namespace sardine
{

    namespace py = pybind11;

    using py_array = py::array;
    // using ret_py_array = py::ndarray<py::numpy>;

namespace detail
{

        template<>
        struct as_span_of_bytes<py_array&> {
            static span_b as_span(py_array& value) {
                return std::span{reinterpret_cast<std::byte*>(value.mutable_data()), value.nbytes()};
            }
        };

        template<>
        struct as_span_of_bytes<const py_array&> {
            static span_cb as_span(const py_array& value) {
                return std::span{reinterpret_cast<const std::byte*>(value.data()), value.nbytes()};
            }
        };

} // namespace detail

    inline result<span_b> bytes_from_url( url_view u, py::type type) {
        // for now, only handle cpu buffer.
        constexpr auto requested_dt = emu::dlpack::device_type_t::kDLCPU; //emu::location_type_of<T>::device_type;

        //here handle cases where opening url need to also know the type
        // but the result is still bytes.
        auto scheme = u.scheme();

        result<bytes_and_device> bytes_and_device;

        //TODO: implement custom function to parse json url into numpy array.
        // if ( scheme == region::embedded::url_scheme )
        //     bytes_and_device = region::embedded::bytes_from_url<T>( u, requested_dt );
        // else
            bytes_and_device = detail::bytes_from_url( u );

        // Once we have the bytes, convert then if necessary.
        return bytes_and_device.and_then(detail::byte_converter_for(requested_dt));
    }

namespace buffer::detail
{

    template <>
    struct adaptor_type < py_array > {
        using interface_type = py_array&;
    };

} // namespace buffer::detail



    inline uint8_t code_from_np_types(int numpy_type) {
        switch (numpy_type) {
            case py::detail::npy_api::constants::NPY_BOOL_:
                return kDLUInt;

            case py::detail::npy_api::constants::NPY_INT8_:
            case py::detail::npy_api::constants::NPY_INT16_:
            case py::detail::npy_api::constants::NPY_INT32_:
            case py::detail::npy_api::constants::NPY_INT64_:
                return kDLInt;

            case py::detail::npy_api::constants::NPY_UINT8_:
            case py::detail::npy_api::constants::NPY_UINT16_:
            case py::detail::npy_api::constants::NPY_UINT32_:
            case py::detail::npy_api::constants::NPY_UINT64_:
                return kDLUInt;

            case py::detail::npy_api::constants::NPY_FLOAT_:
            case py::detail::npy_api::constants::NPY_DOUBLE_:
            case py::detail::npy_api::constants::NPY_LONGDOUBLE_:
                return kDLFloat;

            default:
                return kDLOpaqueHandle;
        }

    }

    inline int dlpack_type_to_numpy(const emu::dlpack::data_type_ext_t& dtype) {
        switch (dtype.code) {
            case kDLInt:
                switch (dtype.bits) {
                    case 8: return py::detail::npy_api::constants::NPY_INT8_;
                    case 16: return py::detail::npy_api::constants::NPY_INT16_;
                    case 32: return py::detail::npy_api::constants::NPY_INT32_;
                    case 64: return py::detail::npy_api::constants::NPY_INT64_;
                    default: throw std::invalid_argument("Unsupported DLInt bit width");
                }

            case kDLUInt:
                switch (dtype.bits) {
                    case 8: return py::detail::npy_api::constants::NPY_UINT8_;
                    case 16: return py::detail::npy_api::constants::NPY_UINT16_;
                    case 32: return py::detail::npy_api::constants::NPY_UINT32_;
                    case 64: return py::detail::npy_api::constants::NPY_UINT64_;
                    default: throw std::invalid_argument("Unsupported DLUInt bit width");
                }

            case kDLFloat:
                switch (dtype.bits) {
                    case 32: return py::detail::npy_api::constants::NPY_FLOAT_;
                    case 64: return py::detail::npy_api::constants::NPY_DOUBLE_;
                    case 128: return py::detail::npy_api::constants::NPY_LONGDOUBLE_;
                    default: throw std::invalid_argument("Unsupported DLFloat bit width");
                }

            case kDLOpaqueHandle:
                // kDLOpaqueHandle doesn't have a direct equivalent in NumPy. You can return a special value.
                // For example, return NPY_VOID or create your own custom enum value for opaque handles.
                return py::detail::npy_api::constants::NPY_VOID_;

            default:
                throw std::invalid_argument("Unsupported DLPack type code");
        }
    }

    inline bool is_strided(const pybind11::array& arr) {
        // Get the strides and shape of the array
        auto strides = arr.strides();
        auto shape = arr.shape();
        auto itemsize = arr.itemsize();

        // Check if the array is contiguous by comparing strides to the expected layout
        size_t expected_stride = itemsize;
        for (ssize_t i = arr.ndim() - 1; i >= 0; --i) {
            if (strides[i] != expected_stride) {
                return true; // The array is strided
            }
            expected_stride *= shape[i];
        }

        // If all strides match the expected contiguous layout, the array is not strided
        return false;
    }

    // This is a fake array pointer that will be used to compute change of the shape
    // of the input array.
    // When the value is 0 or NULL, py::array will reallocate the memory. that make
    // offset invalid. That is why we are adding this offset at the creation
    // and subtracting it afterward.
    constexpr static size_t fake_ptr = 1;

    template<>
    struct mapper< py_array >
    {
        using view_type = py_array;
        // using convert_type = py_array;
        // using container_type = py_array;

        py_array fake_array;
        emu::capsule capsule;

    public:
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

            py_array fake_array(
                py::dtype(dlpack_type_to_numpy(f.data_type())),
                move(shape),
                move(strides),
                reinterpret_cast<void*>(fake_ptr),
                py::str() // dummy parent class to avoid making a copy and reading the fake_ptr.
            );
            return mapper{std::move(fake_array), emu::capsule()};
        }

        static mapper from(const py_array& array) {
            const auto rank = array.ndim();

            std::vector<size_t> shape(rank), strides(rank);
            std::copy_n(array.shape(), rank, shape.begin());
            std::copy_n(array.strides(), rank, strides.begin());

            py_array fake_array(
                array.dtype(),
                move(shape), move(strides),
                reinterpret_cast<void*>(fake_ptr),
                py::str() // dummy parent class to avoid making a copy and reading the fake_ptr
            );

            return mapper{std::move(fake_array), emu::capsule(std::move(array))};
        }

        size_t size() const { return required_span_size(); }
        size_t offset() const { return reinterpret_cast<size_t>(fake_array.data()) - fake_ptr; }
        size_t lead_stride() const { return fake_array.strides(0); } // what about layout_f ?

        size_t required_span_size() const noexcept {
            size_t span_size = 1;
            for(unsigned r = 0; r < fake_array.ndim(); r++) {
                // Return early if any of the extents are zero
                if(fake_array.shape(r)==0) return 0;
                //! stride is probably in bytes.
                span_size = std::max(span_size, static_cast<size_t>(fake_array.shape(r) * fake_array.strides(r)));
            }
            return span_size;
        }

        mapper close_lead_dim() const {
            return submdspan_mapper(py::make_tuple(0));
        }

        py_array convert(span_b buffer) const {
            const auto rank = fake_array.ndim();

            //TODO consider to use a small_vector with SBO.
            std::vector<ssize_t> shape(rank), strides(rank);
            std::copy_n(fake_array.shape(), rank, shape.begin());
            std::copy_n(fake_array.strides(), rank, strides.begin());

            return py_array(
                fake_array.dtype(),
                move(shape), move(strides),
                reinterpret_cast<void*>(buffer.data() + offset()),
                py::str()
            );
        }

        template<typename TT>
        static auto as_bytes(TT&& array) {
            byte* ptr = reinterpret_cast<byte*>(array.mutable_data());

            return emu::as_writable_bytes(std::span{ptr, array.size() * array.itemsize()});
        }


        default_mapping_descriptor mapping_descriptor() const {
            std::vector<size_t> extents; extents.resize(fake_array.ndim());
            for (size_t i = 0; i < fake_array.ndim(); ++i)
                extents[i] = fake_array.shape(i);

            std::vector<size_t> strides;
            //TODO: check if array is strided...
            if (is_strided(fake_array)) {
                strides.resize(fake_array.ndim());
                for (size_t i = 0; i < fake_array.ndim(); ++i)
                    strides[i] = fake_array.strides(i);
            }

            auto dtype = fake_array.dtype();

            return default_mapping_descriptor(
                  /* extents = */ std::move(extents),
                  /* strides = */ std::move(strides),
                  /* data_type = */ emu::dlpack::data_type_ext_t{
                    .code = code_from_np_types(dtype.num()),
                    .bits = dtype.itemsize() * CHAR_BIT,
                    .lanes = 1
                  },
                  /* offset = */ offset(),
                  /* is_const = */ false
            );
        }


        // span_b to_bytes(MdSpan value) const {
        //     return std::as_writable_bytes(value);
        // }

        // void copy(const py_array& src, py_array& dst) const {
        //     //TODO: implement copy that works for mdspan (with strides support)
        //     py::dict scope;
        //     scope["src"] = src; scope["dst"] = dst;

        //     py::exec("dst[()] = src", scope);
        // }

        // void update_json(json::object& jo) const {

        //     jo[extent_key] = json::value_from(mapping.extents()); // or use fmt::format("[{}]", emu::extent(value, ","))

        //     if constexpr (not value.is_always_exhaustive())
        //         if (not value.is_exhaustive())
        //             jo[stride_key] = json::value_from(mapping.strides()); // or use fmt::format("[{}]", emu::stride(value, ","))
        // }

    protected:
        mapper submdspan_mapper(py::handle args) const
        {
            py::dict scope;
            scope["mapping"] = fake_array;
            scope["slices"] = args;

            auto result = py::cast<py_array>(py::eval("mapping[slices]", scope));

            // We have to create a copy of the result shape, result.shape_ptr() is int64_t* and we need size_t*.
            // std::vector<size_t> new_shape(result.ndim());
            // std::copy_n(result.shape_ptr(), result.ndim(), new_shape.begin());

            // auto offset = (size_t)res.data();

            // fmt::print("offset = {}\n", offset / a.itemsize());

            // auto ptr = ((uint8_t*)a.data()) + offset;


            return mapper{result, capsule};

            // fake_array was created with a fake pointer of 1. We need to remove this value.
            // auto offset = reinterpret_cast<size_t>(result.data()) - fake_ptr;

            // Create a fake mdspan from offset and size. Use deduction guide.
            // emu::mdspan fake_mdspan(reinterpret_cast< std::byte* >(offset_), this->mapping_);

            // let mdspan do the computation to get the new submdspan.
            // auto sv = emu::submdspan(fake_mdspan, specs...);

            // using new_mapping_t = typename decltype(sv)::mapping_type;
            // using new_mdspan_t = emu::mdspan<element_type, typename new_mapping_t::extents_type, typename new_mapping_t::layout_type>;

            // return sardine::view_mapper<new_mdspan_t>{reinterpret_cast<size_t>(sv.data_handle()), sv.mapping()};
        }
    };

namespace buffer
{
    template <typename Derived>
    struct mapper_base< py_array, Derived > : sardine::mapper< py_array >
    {
        using mapper_t = sardine::mapper< py_array >;

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
