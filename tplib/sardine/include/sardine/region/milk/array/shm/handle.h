#ifndef CACAO_SHM_HANDLE_H
#define CACAO_SHM_HANDLE_H

#include <cacao/detail/type.h>
#include <cacao/detail/utility.h>

#include <emu/span.h>
#include <emu/scoped.h>

#include <ImageStreamIO.h>
#include <ImageStruct.h>

#if SEMAPHORE_MAXVAL != 1
    #error "SEMAPHORE_MAXVAL must be set to 1"
#endif

namespace cacao::shm
{

    struct handle_t;

    std::string filename(emu::string_cref name);

    bool exists(emu::string_cref name);

    void remove(emu::string_cref name);

    void throw_already_exists(emu::string_cref name);

    void throw_if_exists(emu::string_cref name);

    struct remover {
        std::string name;
        remover(std::string name);
        ~remover();
    };

namespace handle
{

    using id_t = IMAGE;

namespace detail
{

    /// Open an existing image.
    id_t open(emu::string_cref name);

    /// Create an image and connect to it.
    id_t create(emu::string_cref name, emu::span_t<const std::size_t> dims, std::size_t buffer_nb, type_t type, int location);

    /// Create a or connect to an image.
    id_t open_or_create(emu::string_cref name, emu::span_t<const std::size_t> dims, std::size_t buffer_nb, type_t type, int location);

    void close(id_t & handle);

    void remove(id_t & handle);

    struct Destroyer{
        void operator()(id_t& handle) const { close(handle); }
    };

    void throw_if_mismatch(const id_t & handle, emu::span_t<const std::size_t> shape, std::size_t buffer_nb, type_t type, int location);

} // namespace detail

    using ScopedHandle = emu::scoped_t<id_t, detail::Destroyer>;

} // namespace handle

    struct handle_t
    {
        handle_t() = default;

        handle_t(handle::id_t handle, bool owning);

        handle_t(handle_t&&) = default;
        handle_t& operator=(handle_t&&) = default;

        ~handle_t() = default;

        handle::id_t & id() { return id_.value; }
        const handle::id_t & id() const { return id_.value; }

        byte* base_buffer() const;
        byte* current_buffer() const;
        byte* next_buffer() const;

        std::size_t cnt() const;
        std::size_t index() const;
        std::size_t next_index() const;

        std::size_t* cnt_ptr() const;
        std::size_t* index_ptr() const;

        type_t type() const noexcept;
        int location() const noexcept;

        emu::span_t<const std::size_t> shape() const;
        std::size_t buffer_nb() const;
        std::size_t size() const;
        std::size_t size_total() const;

        std::uint64_t wait(int sem_index);
        void notify(std::uint64_t idx);
        bool available(int sem_index) const;

        int lock_sem_index();
        void unlock_sem_index(int sem_index);

        std::string sem_name_at(int sem_idx) const;

        std::string name() const;
        void destroy();

    private:
        handle::ScopedHandle id_;
        std::array<std::size_t, 2> shape_;
    };

namespace handle
{

namespace detail
{
    struct SemIndexUnlocker
    {
        shm::handle_t* handle;

        void operator()(int sem_index)
        {
            handle->unlock_sem_index(sem_index);
        }
    };

} // namespace detail

    using ScopedSemIndex = emu::scoped_t<int, detail::SemIndexUnlocker>;

    handle_t open(emu::string_cref name);

    handle_t create(emu::string_cref name, emu::span_t<const std::size_t> dims, std::size_t buffer_nb, type_t type, int location);

    handle_t open_or_create(emu::string_cref name, emu::span_t<const std::size_t> dims, std::size_t buffer_nb, type_t type, int location);

    handle_t wrap(id_t handle, bool take_ownership);

} // namespace handle

} // namespace cacao::shm

#endif //CACAO_SHM_HANDLE_H