#include <sardine/milk/array.hpp>

namespace sardine::milk
{

    void remove_array(cstring_view name) {
        array::remove(name);
    }

    ArrayBase::ArrayBase(s_image_t h)
        : handle_(h) //, data_type(handle().type())
        , local_index(handle().index())
        , sem_index(handle().lock_sem_index())
        // , sem_name_(handle().sem_name_at(sem_index.value))
    {}

    ArrayBase::ArrayBase(const ArrayBase & array)
        : handle_(array.handle_) //, data_type(array.data_type)
        , local_index(array.local_index)
        // The created ArrayBase share the handle but get an unique semaphore index
        , sem_index(handle().lock_sem_index())
        // , sem_name_(handle().sem_name_at(sem_index.value))
    {}

    ArrayBase& ArrayBase::operator=(const ArrayBase & array)
    {
        handle_ = array.handle_;
        // data_type = array.data_type;
        local_index = array.local_index;
        // The created ArrayBase share the handle but get an unique semaphore index
        sem_index = handle().lock_sem_index();
        // sem_name_ = handle().sem_name_at(sem_index.value);

        return *this;
    }

    auto ArrayBase::handle() -> image_t& {
        return *handle_;
    }

    auto ArrayBase::handle() const -> const image_t& {
        return *handle_;
    }

    void ArrayBase::wait() {
        local_index = handle().wait_on(sem_index.value);
    }

    void ArrayBase::notify() {
        handle().set_index_and_notify(local_index);
    }

    void ArrayBase::next() {
        local_index = handle().next_index();
    }

    void ArrayBase::sync_ptr() {
        local_index = handle().index();
    }

    bool ArrayBase::available() const {
        // return local_index != handle().index();
        return handle().available(sem_index.value);
    }

    uint64_t ArrayBase::current_index() const {
        return local_index;
    }

    uint64_t ArrayBase::next_index() const {
        return handle().next_index();
    }

    size_t ArrayBase::buffer_nb() const {
        return handle().buffer_nb();
    }


    size_t ArrayBase::rank() const {
        return handle().rank();
    }

    size_t ArrayBase::size() const {
        return handle().size();
    }

    size_t ArrayBase::size_byte() const {
        return size() * item_size();
    }

    span<const size_t> ArrayBase::extents() const {
        return handle().extents();
    }

    span_b ArrayBase::base_bytes() const {
        return handle().base_buffer();
    }

    span_b ArrayBase::current_bytes() const {
        return base_bytes().subspan( local_index * size_byte(), size_byte());
    }

    span_b ArrayBase::next_bytes() const {
        return handle().next_buffer();
    }

    int ArrayBase::location() const {
        return handle().location();
    }

    dtype_t ArrayBase::dtype() const {
        return handle().dtype();
    }

    size_t ArrayBase::item_size() const {
        return handle().item_size();
    }


    // cstring_view ArrayBase::sem_name() const {
    //     return sem_name_;
    // }

    cstring_view ArrayBase::name() const {
        return handle().name();
    }

    url ArrayBase::url() const {
        return handle().get_url();
    }

    ArrayBase from_url(url_view u) {
        return {array::image_t::from_url(u)};
    }

    ArrayBase open(string name) {
        return {array::image_t::open(name)};
    }

    ArrayBase create(string name, span<const size_t> shape, size_t buffer_nb, dtype_t dtype, int location, bool overwrite) {
        if (array::exists(name) and overwrite)
            array::remove(name);

        return {array::image_t::create(name, shape, buffer_nb, dtype, location)};
    }

    ArrayBase open_or_create(string name, span<const size_t> shape, size_t buffer_nb, dtype_t dtype, int location, bool overwrite) {
        if (array::exists(name) and overwrite)
            array::remove(name);

        return {array::image_t::open_or_create(name, shape, buffer_nb, dtype, location)};
    }

namespace host
{

    ArrayBase open(string name) {
        auto res = sardine::milk::open(name);
        if (res.location() != -1)
            throw std::runtime_error(fmt::format("Location mismatch: expected {}, got {}", -1, res.location()));

        return res;
    }

    ArrayBase create(string name, span<const size_t> shape, size_t buffer_nb, dtype_t dtype, bool overwrite) {
        return sardine::milk::create(name, shape, buffer_nb, dtype, -1, overwrite);
    }

    ArrayBase open_or_create(string name, span<const size_t> shape, size_t buffer_nb, dtype_t dtype, bool overwrite) {
        return sardine::milk::open_or_create(name, shape, buffer_nb, dtype, -1, overwrite);
    }

} // namespace host

namespace cuda
{

    ArrayBase open(string name) {
        auto res = sardine::milk::open(name);
        if (res.location() == -1)
            throw std::runtime_error(fmt::format("Location mismatch: expected not {}, got {}", -1, res.location()));

        return res;
    }

    ArrayBase create(string name, span<const size_t> shape, size_t buffer_nb, dtype_t dtype, int location, bool overwrite) {
        return sardine::milk::create(name, shape, buffer_nb, dtype, location, overwrite);
    }

    ArrayBase open_or_create(string name, span<const size_t> shape, size_t buffer_nb, dtype_t dtype, int location, bool overwrite) {
        return sardine::milk::open_or_create(name, shape, buffer_nb, dtype, location, overwrite);
    }

} // namespace cuda

} // namespace sardine::milk
