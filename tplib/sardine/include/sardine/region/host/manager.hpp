#pragma once

#include <sardine/type.hpp>
#include <sardine/region/host/utility.hpp>

#include <emu/cstring_view.hpp>
#include <emu/unordered_map.hpp>

namespace sardine::region::host
{

    struct manager : protected emu::unordered_map<string, handle>
    {
        using base = emu::unordered_map<string, handle>;

        static manager &instance();

    private:
        manager() = default;

    public:
        manager(const manager &) = delete;
        manager(manager &&) = delete;

        span_b open(string name);
        span_b create(string name, size_t size);
        span_b open_or_create(string name, size_t size);

        span_b at(cstring_view name);

        // void register_region(string name)

        optional<shm_handle> find_handle(const byte* ptr);
    };

} // namespace sardine::region::host
