#pragma once

#include <sardine/type.hpp>
#include <sardine/region/managed.hpp>

namespace sardine::cache
{

namespace detail
{

    region::managed_t get_managed_cache(size_t required_space);

} // namespace detail

    template<typename T, typename... Args>
    auto request(Args&&... args) -> decltype(auto) {
        size_t required_space = 1024;
        while(true) {
            try{
                // No need to check. anonymous will always be created.
                return detail::get_managed_cache(required_space).create_unchecked<T>(region::managed::anonymous_instance, EMU_FWD(args)...);
            } catch (const region::managed::bad_alloc&) {
                required_space *= 2;
            }
        }
    }

    void clear();

} // namespace sardine::cache
