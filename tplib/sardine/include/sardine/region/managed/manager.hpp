#pragma once

#include <sardine/region/managed.hpp>
#include <sardine/region/managed/utility.hpp>

#include <emu/cstring_view.hpp>
#include <emu/associative_container.hpp>
#include <emu/optional.hpp>

#include <string>
#include <cstddef>

namespace sardine::region::managed
{

    struct manager : protected emu::unordered_map<std::string, shared_memory>
    {
        using base = emu::unordered_map<std::string, shared_memory>;

        static manager& instance();

    private:
        manager() = default;

    public:
        manager(const manager&) = delete;
        manager(manager&&) = delete;

        managed_t open(std::string name);
        managed_t create(std::string name, size_t file_size);
        managed_t open_or_create(std::string name, size_t file_size);

        managed_t at(emu::cstring_view name);

        emu::optional<shm_handle> find_handle(const std::byte* ptr);
    };

} // namespace sardine::region::managed
