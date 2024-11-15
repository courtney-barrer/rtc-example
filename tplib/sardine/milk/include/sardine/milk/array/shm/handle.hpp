#pragma once

#include <sardine/milk/type.hpp>
#include <sardine/milk/error.hpp>

#include <emu/scoped.hpp>
#include <emu/managed_object.hpp>

#include <ImageStreamIO/ImageStreamIO.h>
#include <ImageStreamIO/ImageStruct.h>

#if SEMAPHORE_MAXVAL != 1
    #error "SEMAPHORE_MAXVAL must be set to 1"
#endif

namespace sardine::milk::array
{

    constexpr auto url_scheme = "milk";

    struct image_t;

    string filename(cstring_view name);

    bool exists(cstring_view name);

    void remove(cstring_view name);

    // void throw_already_exists(cstring_view name);

    // void throw_if_exists(cstring_view name);

    // struct remover {
    //     string name;
    //     remover(string name);
    //     ~remover();
    // };

namespace image
{

    using handle_t = IMAGE;

namespace detail
{

    /// Open an existing image.
    result<handle_t> open(cstring_view name);

    /// Create an image and connect to it.
    result<handle_t> create(cstring_view name, span<const size_t> extents, size_t buffer_nb, dtype_t type, int location);

    /// Create a or connect to an image.
    result<handle_t> open_or_create(cstring_view name, span<const size_t> extents, size_t buffer_nb, dtype_t type, int location);

    void close(handle_t & handle);

    void remove(handle_t & handle);

    struct Destroyer {
        void operator()(handle_t& handle) const { close(handle); }
    };

    // void throw_if_mismatch(const handle_t & handle, span<const size_t> shape, size_t buffer_nb, dtype_t type, int location);

    struct SemIndexUnlocker
    {
        image_t* handle;

        void operator()(int sem_index);
    };


} // namespace detail

} // namespace image

    using ScopedHandle = emu::scoped<image::handle_t, image::detail::Destroyer>;

    using ScopedSemIndex = emu::scoped<int, image::detail::SemIndexUnlocker>;

    optional<result<url>> url_from_bytes(span_cb data);

    struct image_t : emu::managed_object<string, image_t>
    {
        using base_t = emu::managed_object<string, image_t>;
        using base_t::s_value_type;
        friend base_t;

        friend optional<result<url>> url_from_bytes(span_cb data);

    protected:
        image_t(image::handle_t handle, bool owning);

    public:

        static s_value_type from_url(url_view name);

        static s_value_type open(cstring_view name);

        static s_value_type create(cstring_view name, span<const size_t> extents, size_t buffer_nb, dtype_t type, int location);

        static s_value_type open_or_create(cstring_view name, span<const size_t> extents, size_t buffer_nb, dtype_t type, int location);

        static s_value_type wrap(image::handle_t  handle, bool take_ownership);

        ~image_t() = default;

        image::handle_t & handle() { return handle_.value; }
        const image::handle_t & handle() const { return handle_.value; }

        span_b base_buffer() const;
        span_b current_buffer() const;
        span_b next_buffer() const;

        size_t cnt() const;
        size_t index() const;
        size_t next_index() const;

        size_t* cnt_ptr() const;
        size_t* index_ptr() const;

        dtype_t dtype() const noexcept;
        int location() const noexcept;

        span<const size_t> extents() const;

        size_t rank() const;
        size_t buffer_nb() const;
        size_t item_size() const;

        size_t size() const;
        size_t size_total() const;

        size_t size_bytes() const;
        size_t size_total_bytes() const;

        uint64_t wait_on(int sem_index);
        void set_index_and_notify(uint64_t new_index);
        bool available(int sem_index) const;

        ScopedSemIndex lock_sem_index();
        void unlock_sem_index(int sem_index);

        string sem_name_at(int sem_idx) const;

        string name() const;
        void destroy();

        sardine::url get_url() const;

    private:
        ScopedHandle handle_;
        std::array<size_t, 2> extents_;
    };

    using s_image_t = image_t::s_value_type;

} // namespace sardine::milk::array
