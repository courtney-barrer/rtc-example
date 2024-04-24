#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <span>

namespace detail
{

    template<typename T>
    void* void_ptr(T* ptr) {
        return reinterpret_cast<void*>(ptr);
    }

    template<typename T>
    const void* void_ptr(const T* ptr) {
        return reinterpret_cast<const void*>(ptr);
    }

} // namespace detail


namespace nanobind::detail
{

    /**
     * @brief Type caster for std::span<T> <-> numpy array.
     *
     * This type caster transform back and forth between a span and a numpy array.
     *
     * Note: A type cast requires c++ type to be default constructible. This is not the case for span with static extent.
     *
     * @tparam T that data type
     */
    template<typename T>
    struct type_caster< std::span<T> > {
        NB_TYPE_CASTER(std::span<T>, const_name("span<") + const_name<T>() + const_name(">"))

        // Specify numpy framework both for input and output type. It is fine because nanobind ignore this parameter
        // when casting from Python to C++. It means that we can cast from any framework.
        using nd_array_type = ndarray<T, numpy, ndim<1>, c_contig, device::cpu>;
        using internal_caster = type_caster<nd_array_type>;

        internal_caster caster;

        bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
            const auto no_convert = ~ static_cast<uint8_t>(cast_flags::convert);

            // Try to cast to ndarray, span are non owning so we want to be sure to use the original data (no conversion).
            bool res = caster.from_python(src, flags & no_convert, cleanup);
            if (res)
                // If successful, convert to span.
                value = std::span<T>(caster.value.data(), caster.value.size());

            return res;
        }

        static handle from_cpp(std::span<T> src, rv_policy, cleanup_list *cleanup) noexcept {
            // Create a non owning ndarray from the span and cast it to Python handle.
            return internal_caster::from_cpp(
                nd_array_type(::detail::void_ptr(src.data()), {src.size()}),
                rv_policy::reference, cleanup
            );
        }
    };

} // namespace nanobind::detail
