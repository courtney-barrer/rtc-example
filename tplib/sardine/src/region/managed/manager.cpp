#include <sardine/region/managed/manager.hpp>

#include <sardine/type.hpp>
#include <sardine/utility.hpp>

#include  <stdexcept>

namespace sardine::region::managed
{

    manager& manager::instance() {
        static manager obj;
        return obj;
    }

    managed_t manager::open(std::string name) {
        return managed_t{&find_or_emplace(static_cast<base&>(*this), name, [name] {
            return shared_memory(open_only, name.c_str());
        })->second};
    }

    managed_t manager::create(std::string name, size_t file_size) {
        return managed_t{&emplace_or_throw(static_cast<base&>(*this), name, [name, file_size] {
            return shared_memory(create_only, name.c_str(), file_size);
        })->second};
    }

    managed_t manager::open_or_create(std::string name, size_t file_size) {
        return managed_t{&find_or_emplace(static_cast<base&>(*this), name, [name, file_size] {
            return shared_memory(sardine::open_or_create, name.c_str(), file_size);
        })->second};
    }

    managed_t manager::at(cstring_view name) {
        if (auto it = base::find(name); it != base::end())
            return managed_t{&it->second};
        else
            throw std::out_of_range(fmt::format("No shared memory with name '{}'", name));
    }

    optional<shm_handle> manager::find_handle(const std::byte* ptr) {
        for (auto& [name, shm] : *this) {
            if (shm.belongs_to_segment(ptr))
                return shm_handle{emu::cstring_view(name), {&shm}, shm.get_handle_from_address(ptr)};
        }
        return nullopt;
    }


} // namespace sardine::region::managed
