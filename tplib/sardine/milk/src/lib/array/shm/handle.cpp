
#include <sardine/milk/array/shm/handle.hpp>
#include <sardine/milk/error.hpp>
#include <sardine/milk/utility.hpp>

#include <sardine/region_register.hpp>
#include <sardine/mapping_descriptor/base.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <filesystem>
#include <sched.h>

namespace sardine::milk::array
{

    string filename(cstring_view name)
    {
        char res[400];

    MILK_CHECK_THROW_LOG(ImageStreamIO_filename(res, 400, name.c_str()),
            "Cannot retrieve shm filename for {}", name);

        return {res};
    }

    bool exists(cstring_view name)
    {
        std::error_code ec;

        auto res = std::filesystem::exists(filename(name), ec);

        if (ec)
            emu::throw_error(ec);

        return res;
    }

    void remove(cstring_view name)
    {
        std::error_code ec;

        std::filesystem::remove(filename(name), ec);

        if (ec)
            emu::throw_error(ec);
    }

    // remover::remover(string name):name(name) {
    //     shm::remove(name);
    // }

    // remover::~remover() {
    //     shm::remove(name);
    // }

namespace image
{

namespace detail
{

    std::array<uint32_t, 3> dimensions(span<const size_t> extents, size_t buffer_nb) {
        if (extents.size() != 2)
            throw std::runtime_error(fmt::format("Expect 2 dimensions extents, got {} with {} dimensions", extents, extents.size()));

        return {static_cast<uint32_t>(extents[0]), static_cast<uint32_t>(extents[1]), static_cast<uint32_t>(buffer_nb)};
    }

    dtype_t dtype(const handle_t& handle) {
        return sardine::milk::to_dtype(handle.md->datatype);
    }

    std::array<size_t, 2> extents(const handle_t & handle) {
        return {handle.md->size[0], handle.md->size[1]};
    }


    size_t rank(const handle_t & handle) {
        // When the image is 3D, the third dimension is the buffer number. So we return 2.
        return (handle.md->naxis == 3 ? 2 : handle.md->naxis);
    }

    size_t buffer_nb(const handle_t & handle) {
        return ImageStreamIO_nbSlices(&handle);
    }

    int location(const handle_t & handle) {
        return handle.md->location;
    }

    string name(const handle_t & handle) {
        return {handle.name};
    }

    bool type_match(const handle_t & handle, dtype_t requested_type) {
        return dtype(handle) == requested_type;
    }

    bool shape_match(const handle_t & handle, span<const size_t> s, size_t buffer_nb) {
        auto h_s = extents(handle);
        return (h_s.size() == s.size()) and std::equal(s.begin(), s.end(), h_s.begin()) and (buffer_nb == rank(handle));
    }

    bool location_match(const handle_t & handle, int l) {
        return location(handle) == l;
    }

    // void throw_type_mismatch(const handle_t & handle, dtype_t requested_type) {
    //     throw std::runtime_error(fmt::format("Type mismatch: expected {}, got {}", requested_type, type(handle)));
    // }

    // void throw_shape_mismatch(const handle_t & handle, span<const size_t> s, size_t buffer_nb) {
    //     throw std::runtime_error(fmt::format("Shape mismatch: expected {}x{}, got {}x{}", s, buffer_nb, extents(handle), rank(handle)));
    // }

    // void throw_location_mismatch(const handle_t & handle, int l) {
    //     throw std::runtime_error(fmt::format("Location mismatch: expected {}, got {}", l, location(handle)));
    // }

    result<void> check_if_mismatch(const handle_t & handle, span<const size_t> extents, size_t buffer_nb, dtype_t dtype, int location) {

        EMU_TRUE_OR_RETURN_UN_EC_LOG(type_match(handle, dtype), errc::array_open_type_mismatch,
            "Type mismatch: expected {}, got {}", dtype, detail::dtype(handle));
        EMU_TRUE_OR_RETURN_UN_EC_LOG(shape_match(handle, extents, buffer_nb), errc::array_open_shape_mismatch,
            "Shape mismatch: expected {}x{}, got {}x{}", extents, buffer_nb, detail::extents(handle), detail::buffer_nb(handle));
        EMU_TRUE_OR_RETURN_UN_EC_LOG(location_match(handle, location), errc::array_open_location_mismatch,
            "Location mismatch: expected {}, got {}", location, detail::location(handle));

        return {};
    }

    result<handle_t> open(cstring_view name)
    {
        handle_t handle;

        // fmt::print("Open shm {}\n", name);

        MILK_CHECK_RETURN_UN_EC_LOG(ImageStreamIO_read_sharedmem_image_toIMAGE(name.c_str(), &handle),
            "Fail to connect to {}", name);

        return handle;
    }

    result<handle_t> create(cstring_view name, span<const size_t> extents, size_t rank, dtype_t type, int location)
    {
        handle_t handle;

        // fmt::print("Creat shm at {}\n", name);

        // throw_if_exists(name);

        // fmt::print("shm {} did not exists, creating...\n", name);

        auto dims = dimensions(extents, rank);

        MILK_CHECK_RETURN_UN_EC_LOG(ImageStreamIO_createIm_gpu(
            /* image = */     &handle,
            /* name = */      name.c_str(),
            /* naxis = */     dims.size(),
            /* size = */      dims.data(),
            /* datatype = */  EMU_UNWRAP(to_image_type(type)),
            /* location = */  location,
            /* shared = */    1,
            /* NBsem = */     IMAGE_NB_SEMAPHORE,
            /* NBkw = */      0, // ?? was 1 before, maybe revert if issue
            /* imagetype = */ CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
            /* CBsize = */    0
        ), "Fail to create {}", name);

        return handle;
    }

    result<handle_t> open_or_create(cstring_view name, span<const size_t> extents, size_t rank, dtype_t type, int location)
    {
        if(exists(name)) {
            // Do not unwrap it here to avoid having to wrap it again when returning.
            auto res = open(name);

            if (res) {
                EMU_RES_TRUE_OR_RETURN_UN_EC_LOG(check_if_mismatch(*res, extents, rank, type, location),
                    "Mismatch for {}", name);
            }

            return res;
        }
        else
            return create(name, extents, rank, type, location);
    }

    void close(handle_t & handle)
    {
        // fmt::print("Close shm {}\n", name(handle));
        ImageStreamIO_closeIm(&handle);
    }

    void destroy(handle_t & handle)
    {
        // fmt::print("Destroy shm {}\n", name(handle));
        ImageStreamIO_destroyIm(&handle);
    }

    bool is_proccess_alive(pid_t pid)
    {
        auto saved_errno = errno;

        auto res = kill(pid, 0);
        EMU_ASSERT_MSG(errno != EPERM, "Cannot check if process is alive");

        errno = saved_errno;

        return res;
    }

    int lock_sem_index(handle_t & handle)
    {
        // auto res = -1;

        // TODO need to investigate and check if we can still use ImageStreamIO_getsemwaitindex.
        // right now, the issue is that ImageStreamIO only allow one semaphore by shm by process.
        // But cacao new model rely on having one semaphore per instance of handle.
        // Maybe try to find another way.
        for (int semindex = 0; semindex < handle.md->sem; ++semindex)
        {
            // getpgid returns -1 if the process does not exist.
            if ((handle.semReadPID[semindex] == 0) ||(getpgid(handle.semReadPID[semindex]) < 0))
            {
                handle.semReadPID[semindex] = getpid();
                return semindex;
            }
        }
        // auto res = ImageStreamIO_getsemwaitindex(&handle, -1);
        // if (res == -1)
        throw std::runtime_error(fmt::format("Cannot lock semaphore for {}", name(handle)));
        // return res;
    }

    void unlock_sem_index(handle_t & handle, int index)
    {
        handle.semReadPID[index] = 0;
    }

    void SemIndexUnlocker::operator()(int sem_index)
    {
        handle->unlock_sem_index(sem_index);
    }

} // namespace detail

} // namespace image

    auto image_t::from_url(url_view name) -> s_value_type
    {
        auto path = name.path().substr(1); // Remove the first slash.

        return image_t::open(path);
    }


    auto image_t::open(cstring_view name) -> s_value_type
    {
        return find_or_emplace(name, [name = name]{
            return new image_t(image::detail::open(name).value(), true);
        }).first;
    }

    auto image_t::create(cstring_view name, span<const size_t> extents, size_t buffer_nb, dtype_t type, int location) -> s_value_type
    {
        return emplace_or_throw(name, [&, name = name]{
            return new image_t(image::detail::create(name, extents, buffer_nb, type, location).value(), true);
        }).first;
    }

    auto image_t::open_or_create(cstring_view name, span<const size_t> extents, size_t buffer_nb, dtype_t type, int location) -> s_value_type
    {
        auto res = find_or_emplace(name, [&, name = name]{
            return new image_t(image::detail::create(name, extents, buffer_nb, type, location).value(), true);
        });

        if (not res.second){ // If the image already exists.
            EMU_RES_TRUE_OR_THROW_LOG(image::detail::check_if_mismatch(res.first->handle(), extents, buffer_nb, type, location),
                    "Mismatch for {}", name);
        }

        return res.first;
    }

    auto image_t::wrap(image::handle_t handle, bool take_ownership) -> s_value_type
    {
        return try_emplace(image::detail::name(handle), handle, take_ownership).first;
    }


    image_t::image_t(image::handle_t handle, bool owning) :
        handle_(handle, owning),
        extents_(image::detail::extents(handle_.value))
    {}

    span_b image_t::base_buffer() const
    {
        return {reinterpret_cast<byte*>(handle().array.raw), size_total_bytes()};
    }

    span_b image_t::current_buffer() const
    {
        byte *buffer;
        ImageStreamIO_readLastWroteBuffer(&handle(), reinterpret_cast<void**>(&buffer));
        return {buffer, size_bytes()};
    }

    span_b image_t::next_buffer() const
    {
        byte *buffer;
        ImageStreamIO_writeBuffer(&handle(), reinterpret_cast<void**>(&buffer));
        return {buffer, size_bytes()};
    }

    size_t image_t::cnt() const
    {
        return handle().md->cnt0;
    }

    size_t image_t::index() const
    {
        return handle().md->cnt1;
    }

    size_t image_t::next_index() const
    {
        return ImageStreamIO_writeIndex(&handle());
    }

    size_t* image_t::cnt_ptr() const
    {
        return &handle().md->cnt0;
    }

    size_t* image_t::index_ptr() const
    {
        return &handle().md->cnt1;
    }

    dtype_t image_t::dtype() const noexcept
    {
        return image::detail::dtype(handle());
    }

    int image_t::location() const noexcept
    {
        return handle().md->location;
    }

    span<const size_t> image_t::extents() const
    {
        return span{extents_}.subspan(0, rank());
    }

    size_t image_t::rank() const
    {
        return image::detail::rank(handle());
    }

    size_t image_t::buffer_nb() const
    {
        return image::detail::buffer_nb(handle());
    }

    size_t image_t::item_size() const
    {
        return dtype().bits / CHAR_BIT;
    }

    size_t image_t::size() const
    {
        return handle().md->size[0] * handle().md->size[1];
    }

    size_t image_t::size_total() const
    {
        return size() * buffer_nb();
    }

    size_t image_t::size_bytes() const
    {
        return size() * item_size();
    }

    size_t image_t::size_total_bytes() const
    {
        return size_total() * item_size();
    }

    uint64_t image_t::wait_on(int sem_index)
    {
        // No need to check in release mode, as the precondition is already checked.
        EMU_VERIFY(ImageStreamIO_semwait(&handle(), sem_index) == 0);

        // Maybe adds a memory barrier here to ensure that the read is done after the semaphore is taken.
        return index();
    }

    void image_t::set_index_and_notify(uint64_t new_index)
    {
        // Increment update counter.
        handle().md->cnt0++;

        *index_ptr() = new_index;

        // No need to check the return value here, as it is always 0.
        ImageStreamIO_sempost(&handle(), -1);
    }

    bool image_t::available(int sem_index) const
    {
        int semaphore_count;

        EMU_TRUE_OR_THROW_ERRNO(sem_getvalue(handle().semptr[sem_index], &semaphore_count) == 0);

        return semaphore_count != 0;
    }

    ScopedSemIndex image_t::lock_sem_index()
    {
        return {image::detail::lock_sem_index(handle()), image::detail::SemIndexUnlocker{this}};
    }

    void image_t::unlock_sem_index(int sem_index)
    {
        image::detail::unlock_sem_index(handle(), sem_index);
    }

    string image_t::sem_name_at(int sem_idx) const {
        return fmt::format(".milk.shm.{}_sem{}", name(), sem_idx);
    }

    string image_t::name() const
    {
        return image::detail::name(handle());
    }

    void image_t::destroy()
    {
        image::detail::destroy(handle());
        // Keep object but will not invoke destructor anymore.
        handle_.reset(handle_.release(), /*owning =*/ false);
    }

    url image_t::get_url() const
    {
        auto u = sardine::url{fmt::format("milk://array/{}", name())};

        std::vector<size_t> extents(1, buffer_nb());
        for (auto e : extents_) extents.push_back(e);

        default_mapping_descriptor dmd{
            /* extents = */ extents,
            /* strides = */ {},
            /* data_type = */ dtype(),
            /* offset = */ 0,
            /* is_const = */ false
        };

        update_url(u, dmd);

        return u;
    }

    optional<result<url>> url_from_bytes(span_cb data) {
        auto& [map, mutex] = image_t::ressource();
        std::lock_guard lg(mutex);

        for (auto& [name, handle] : map) {
            auto s_ptr = handle.lock();
            if (is_contained(s_ptr->base_buffer(), data)) {
                //TODO: add offset and size to the url
                return s_ptr->get_url();
            }
        }

        return emu::nullopt;

    }

    result<bytes_and_device> bytes_from_url(url_view u) {
        auto image = image_t::from_url(u);

        // Take the region here before the move.
        auto region = image->base_buffer();

        return bytes_and_device{
            .region=region,
            .data=region,
            .capsule=emu::capsule(std::move(image))
        };
    }

} // namespace sardine::milk::array

SARDINE_REGISTER_URL_CONVERTER(sardine_milk_url_converter, sardine::milk::array::url_scheme,
    sardine::milk::array::url_from_bytes,
    sardine::milk::array::bytes_from_url
)
