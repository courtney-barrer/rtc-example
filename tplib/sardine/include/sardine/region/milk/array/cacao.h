#ifndef CACAO_CACAO_H
#define CACAO_CACAO_H

#include <cacao/detail/utility.h>
#include <cacao/shm/handle.h>
#include <cacao/shm/handle_managed.h>

#include <cstddef>

namespace cacao
{

    using shm::handle::managed::s_handle_t;

    struct Array
    {

        Array() = default;

        Array(s_handle_t h);

        Array(Array && array) = default;
        Array(const Array & array);

        Array& operator=(Array && array) = default;
        Array& operator=(const Array & array);

        ~Array() = default;

        /// Number of buffer
        std::size_t buffer_nb() const;
        /// Number of element in a buffer.
        std::size_t size() const;
        /// Size in byte in a buffer.
        std::size_t size_byte() const;

        /// Shape of a buffer.
        emu::span_t<const std::size_t> shape() const;

        /// Pointer to the first buffer.
        std::byte* base_byte() const;
        /// Pointer to the active buffer.
        std::byte* byte() const;

        template<typename T> T* base_ptr() const { return reinterpret_cast<T*>(base_byte()); }
        template<typename T> T*      ptr() const { return reinterpret_cast<T*>(     byte()); }

        // byte* guard() const;

        // std::uint64_t* cnt_ptr() const;
        // std::uint64_t* index_ptr() const;

        void wait();

        /// Increment shared index and sem notify.
        /// Also increment local if behind.
        void notify();

        /// Increment local index.
        void next();
        /// Assign global index to local one.
        void sync_ptr();

        std::uint64_t index() const;

        /// Check if local is behind.
        bool available() const;

        type_t type() const;
        std::size_t data_size() const;

        /// Return data location
        int location() const;

        shm::handle_t& handle();
        const shm::handle_t& handle() const;

        const char* sem_name() const;
        std::string name() const;

    private:
        s_handle_t handle_;
        type_t data_type;

        std::uint64_t local_index;
        shm::handle::ScopedSemIndex sem_index;
        std::string sem_name_;
    };


    Array open(emu::string_cref name);

    Array create(emu::string_cref name, emu::span_t<const std::size_t> shape, std::size_t buffer_nb, type_t type, int location);

    Array open_or_create(emu::string_cref name, emu::span_t<const std::size_t> shape, std::size_t buffer_nb, type_t type, int location);


namespace host
{

    Array open(emu::string_cref name);

    Array create(emu::string_cref name, emu::span_t<const std::size_t> shape, std::size_t buffer_nb, type_t type);

    Array open_or_create(emu::string_cref name, emu::span_t<const std::size_t> shape, std::size_t buffer_nb, type_t type);

} // namespace host

namespace cuda
{

    Array open(emu::string_cref name);

    Array create(emu::string_cref name, emu::span_t<const std::size_t> shape, std::size_t buffer_nb, type_t type, int location);

    Array open_or_create(emu::string_cref name, emu::span_t<const std::size_t> shape, std::size_t buffer_nb, type_t type, int location);

} // namespace cuda

} // namespace cacao

#endif // CACAO_CACAO_H
