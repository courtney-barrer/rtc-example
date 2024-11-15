#pragma once
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <sardine/utility.hpp>
#include <sardine/mapper/base.hpp>
#include <sardine/mapper/mapper_base.hpp>
#include <sardine/buffer/base.hpp>
#include <sardine/python/dtype.hpp>

#include <emu/pybind11.hpp>
#include <emu/pybind11/cast/detail/capsule.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

#include <span>
#include <type_traits>

namespace sardine
{

    namespace py = pybind11;

    using np_ndarray = py::array;

    inline bool is_strided(const np_ndarray& arr) {
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
    struct mapper< np_ndarray >
    {
        using view_type = np_ndarray;
        // using convert_type = np_ndarray;
        // using container_type = np_ndarray;

        np_ndarray fake_array;
        emu::capsule capsule;

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

        static mapper from(const np_ndarray& array) {
            const auto rank = array.ndim();

            std::vector<size_t> shape(rank), strides(rank);
            std::copy_n(array.shape(), rank, shape.begin());
            std::copy_n(array.strides(), rank, strides.begin());

            np_ndarray fake_array(
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

        np_ndarray convert(span_b buffer) const {
            const auto rank = fake_array.ndim();

            //TODO consider to use a small_vector with SBO.
            std::vector<ssize_t> shape(rank), strides(rank);
            std::copy_n(fake_array.shape(), rank, shape.begin());
            std::copy_n(fake_array.strides(), rank, strides.begin());

            return np_ndarray(
                fake_array.dtype(),
                move(shape), move(strides),
                reinterpret_cast<void*>(buffer.data() + offset()),
                emu::pybind11::detail::capsule_to_handle(capsule)
            );
        }

        template<typename TT>
        static auto as_bytes(TT&& array) {
            byte* ptr = reinterpret_cast<byte*>(array.mutable_data());

            return emu::as_writable_bytes(std::span{ptr, static_cast<size_t>(array.size())});
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
                    .bits = static_cast<uint64_t>(dtype.itemsize()) * CHAR_BIT,
                    .lanes = 1
                  },
                  /* offset = */ offset(),
                  /* is_const = */ false
            );
        }

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
    struct mapper_base< np_ndarray, Derived > : sardine::mapper< np_ndarray >
    {
        using mapper_t = sardine::mapper< np_ndarray >;

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
