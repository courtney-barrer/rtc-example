#pragma once

#include <sardine/milk/type.hpp>
#include <sardine/milk/array/shm/handle.hpp>

#include <string>
#include <cstddef>
#include <span>

namespace sardine::milk
{

    void remove_array(cstring_view name);

    struct ArrayBase
    {
        using image_t = array::image_t;
        using s_image_t = array::s_image_t;
        using ScopedSemIndex = array::ScopedSemIndex;

        ArrayBase() = default;

        ArrayBase(s_image_t h);

        ArrayBase(ArrayBase && array) = default;
        ArrayBase(const ArrayBase & array);

        ArrayBase& operator=(ArrayBase && array) = default;
        ArrayBase& operator=(const ArrayBase & array);

        ~ArrayBase() = default;

        array::image_t& handle();
        const array::image_t& handle() const;

        s_image_t handle_ptr() const { return handle_; }

        void wait();

        /// Increment shared index and sem notify.
        /// Also increment local if behind.
        void notify();

        /// Increment local index.
        void next();
        /// Assign global index to local one.
        void sync_ptr();

        /// Check if local is behind.
        bool available() const;

        uint64_t current_index() const;
        uint64_t next_index() const;

        size_t distance() const {
            auto ci = current_index();
            auto ni = next_index();
            if (ci <= ni) {
                return ni - ci;
            } else {
                return ni + buffer_nb() - ci;
            }
        }


        size_t buffer_nb() const;
        /// Number of buffer
        size_t rank() const;
        /// Number of element in a buffer.
        size_t size() const;
        /// Size in byte in a buffer.
        size_t size_byte() const;

        /// Shape of a buffer.
        span<const size_t> extents() const;


        /// Pointer to the first buffer.
        span_b base_bytes() const;
        /// Pointer to the active local buffer.
        span_b current_bytes() const;
        /// Pointer to the next global buffer.
        span_b next_bytes() const;

        dtype_t dtype() const;
        size_t item_size() const;

        /// Return data location
        int location() const;

        // cstring_view sem_name() const;
        cstring_view name() const;

        sardine::url url() const;

    private:
        s_image_t handle_;
        // type_t data_type;

        uint64_t local_index;
        ScopedSemIndex sem_index;
        // string sem_name_;
    };

    template<typename T>
    struct Array : ArrayBase
    {
        using ArrayBase::ArrayBase;

        span<T> base_view() const { return emu::as_t<T>(base_bytes()); }
        span<T> current_view() const { return emu::as_t<T>(current_bytes()); }
        span<T> next_view() const { return emu::as_t<T>(next_bytes()); }
    };

    ArrayBase from_url(url_view u);

    ArrayBase open(string name);

    ArrayBase create(string name, span<const size_t> extents, size_t buffer_nb, dtype_t type, int location, bool overwrite = false);

    ArrayBase open_or_create(string name, span<const size_t> extents, size_t buffer_nb, dtype_t type, int location, bool overwrite = false);

    template<typename T>
    Array<T> open(string name) {
        auto array = open(name);

        if (array.dtype() != emu::dlpack::data_type_ext<T>) {
            throw std::runtime_error("Array type mismatch.");
        }

        return {std::move(array)};
    }

    template<typename T>
    Array<T> create(string name, span<const size_t> extents, size_t buffer_nb, int location, bool overwrite = true) {
        return {create(name, extents, buffer_nb, emu::dlpack::data_type_ext<T>, location)};
    }

    template<typename T>
    Array<T> open_or_create(string name, span<const size_t> extents, size_t buffer_nb, int location, bool overwrite = true) {
        return {open_or_create(name, extents, buffer_nb, emu::dlpack::data_type_ext<T>, location)};
    }


namespace host
{

    ArrayBase open(string name);

    ArrayBase create(string name, span<const size_t> extents, size_t buffer_nb, dtype_t type, bool overwrite = true);

    ArrayBase open_or_create(string name, span<const size_t> extents, size_t buffer_nb, dtype_t type, bool overwrite = true);

    template<typename T>
    Array<T> open(string name) {
        return {open(name)};
    }

    template<typename T>
    Array<T> create(string name, span<const size_t> extents, size_t buffer_nb, bool overwrite = true) {
        return {create(name, extents, buffer_nb, emu::dlpack::data_type_ext<T>)};
    }

    template<typename T>
    Array<T> open_or_create(string name, span<const size_t> extents, size_t buffer_nb, bool overwrite = true) {
        return {open_or_create(name, extents, buffer_nb, emu::dlpack::data_type_ext<T>)};
    }

} // namespace host

namespace cuda
{

    ArrayBase open(string name);

    ArrayBase create(string name, span<const size_t> extents, size_t buffer_nb, dtype_t type, int location, bool overwrite = true);

    ArrayBase open_or_create(string name, span<const size_t> extents, size_t buffer_nb, dtype_t type, int location, bool overwrite = true);

    template<typename T>
    Array<T> open(string name) {
        return {open(name)};
    }

    template<typename T>
    Array<T> create(string name, span<const size_t> extents, size_t buffer_nb, int location, bool overwrite = true) {
        return {create(name, extents, buffer_nb, emu::dlpack::data_type_ext<T>, location)};
    }

    template<typename T>
    Array<T> open_or_create(string name, span<const size_t> extents, size_t buffer_nb, int location, bool overwrite = true) {
        return {open_or_create(name, extents, buffer_nb, emu::dlpack::data_type_ext<T>, location)};
    }

} // namespace cuda

} // namespace sardine::milk
