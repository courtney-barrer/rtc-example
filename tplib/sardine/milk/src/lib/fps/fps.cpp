
#include <fps/fps.hpp>

#include <ranges>

namespace fps
{

    Object Object::operator[](key_t key) const
    {
        return at(key);
    }

    Object Object::operator[](const char *key) const
    {
        return at(key);
    }

    Object Object::at(key_t key) const
    {
        return {handle_ptr, add_suffix(key)};
    }

    object_iterator Object::begin() const
    {
        auto lvl = level() + 1;
        // `begin` look for the first valid token. `emu::nullopt` (same as end) otherwise.
        return object_iterator(*this, handle().next_token(key, lvl, emu::nullopt), lvl);
    }

    object_iterator Object::end() const
    {
        return {*this, emu::nullopt, level() + 1};
    }

    // flat_object_iterator Object::fbegin() const {
    //     auto lvl = level() + 1;
    //     // `begin` look for the first valid token. `emu::nullopt` (same as end) otherwise.
    //     return flat_object_iterator(*this, handle().next_id(key, lvl, 0), lvl);
    // }
    // flat_object_iterator Object::fend()   const {
    //     return {*this, -1, level() + 1};
    // }

    std::size_t Object::size() const
    {
        return std::distance(begin(), end());
    }

    std::size_t Object::level() const
    {
        return shm::handle::detail::level(key);
    }

    bool Object::has_value() const
    {
        return handle_ptr->has_parameter(key);
    }

    type_t Object::type() const
    {
        if (not has_value())
            shm::throw_field_does_not_exist(handle().name(), key);
        return unchecked_type();
    }

    std::string Object::name() const
    {
        return key.substr(key.find_last_of(".") + 1);
    }

    std::string Object::full_name() const
    {
        return key;
    }

    Object::operator bool() const
    {
        return has_value();
    }

    std::string Object::add_suffix(key_t o_key) const
    {
        // separator is a `.` (dot) only if none of the keys are empty.
        auto separator = (not key.empty() and not o_key.empty()) ? "." : "";

        return key + separator + o_key;
    }

    void Object::init(emu::string_cref desc, type_t type) const
    {
        parameter_ = &handle_ptr->init(key, desc, type);
    }

    type_t Object::unchecked_type() const
    {
        return fps::type(parameter().type);
    }

    shm::handle_t &Object::handle()
    {
        return *handle_ptr;
    }

    shm::handle::parameter_t &Object::parameter() const
    {
        // If parameter_ is not set yet. We
        if (not parameter_)
            parameter_ = &handle_ptr->parameter(key);
        return *parameter_;
    }

    const shm::handle_t &Object::handle() const
    {
        return *handle_ptr;
    }

    Context::Context(s_handle_t h) : // Set temporarily `Object::handle_ptr` to `nullptr` since
                                     // accessing addr of `handle_` is probably undefined behavior.
                                     Object{nullptr, ""}, handle_(emu::mv(h))
    {
        handle_ptr = handle_.get();
    }

    int Context::status() const { return handle().id().md->status; }

    void Context::set_status(int status) { handle().id().md->status = status; }

    Context open(emu::string_cref name)
    {
        using namespace shm::handle;
        return {managed::open(name)};
    }

    Context create(emu::string_cref name)
    {
        using namespace shm::handle;
        return {managed::create(name)};
    }

    Context open_or_create(emu::string_cref name)
    {
        using namespace shm::handle;
        return {managed::open_or_create(name)};
    }

} // namespace fps
