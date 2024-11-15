#include <fps/shm/handle_managed.hpp>
#include <fps/shm/handle.hpp>

#include <fmt/core.h>

#include <unordered_map>

namespace fps
{

namespace shm
{

namespace handle
{

namespace managed
{

    using managed_handle = std::unordered_map<std::string, w_handle_t>;

    managed_handle& ressource() {
        static managed_handle mh;
        return mh;
    }

    bool try_unregister(emu::string_cref name) noexcept {
        auto& mh = ressource();

        return mh.erase(name) == 1;
    }

namespace detail
{

    struct managed_deleter : std::default_delete<handle_t> {

        using base_t = std::default_delete<handle_t>;

        std::string name;

        managed_deleter(std::string name) : base_t(), name(name) {}

        void operator()(handle_t* ptr) const {
            try_unregister(name);
            // fmt::print("Removing {}\n", name);
            base_t::operator()(ptr);
        }
    };

} // namespace detail


    template<typename Finded, typename Emplaced>
    s_handle_t find_or_emplace(emu::string_cref name, Finded finded, Emplaced emplaced) {
        auto& mh = ressource();

        auto it = mh.find(name);
        s_handle_t res;

        // Test if handle is in map.
        if (it != mh.end()) {
            // Convert from weak to shared ptr.
            res = it->second.lock();
        }

        if (res)
            finded(*res);
        else {
            // If not found or fail to convert, invoke function.
            res = emplaced();
            mh.emplace(name, res);
        }

        return res;
    }

    s_handle_t open(emu::string_cref name) {
        return find_or_emplace(name, [](const auto & ) { /* do nothing */ },
            [&name]() -> s_handle_t {
                return {new auto(handle::open(name)), detail::managed_deleter{name}};
            }
        );
    }

    s_handle_t create(emu::string_cref name) {
        return find_or_emplace(name,
            [&](auto & handle) {
                // Create required shm to not exist at all. Throw otherwise.
                throw_if_exists(name);

                handle = handle::create(name);
            },
            [&]() -> s_handle_t {
                return {new auto(handle::create(name)), detail::managed_deleter{name}};
            }
        );
    }

    s_handle_t open_or_create(emu::string_cref name) {
        return find_or_emplace(name, [](const auto & ) { /* do nothing */ },
            [&]() -> s_handle_t {
                return {new auto(handle::open_or_create(name)), detail::managed_deleter{name}};
            }
        );
    }

} // namespace managed

} // namespace handle

} // namespace shm

} // namespace fps
