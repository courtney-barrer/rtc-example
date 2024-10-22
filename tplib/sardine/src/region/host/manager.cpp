#include <sardine/region/host/manager.hpp>

#include <sardine/utility.hpp>

#include <fmt/format.h>

namespace sardine::region::host
{

    manager &manager::instance() {
        static manager instance;
        return instance;
    }

    span_b manager::open(string name) {
        emu::cstring_view name_view = name;
        auto& h = find_or_emplace(static_cast<base&>(*this), std::move(name), [s = name_view.c_str()] {
            auto res = boost::interprocess::shared_memory_object(boost::interprocess::open_only, s, boost::interprocess::read_write);
            return handle(res, boost::interprocess::read_write);
        })->second;

        return map(h);
    }

    span_b manager::create(string name, size_t size) {
        emu::cstring_view name_view = name;
        auto& h = emplace_or_throw(static_cast<base&>(*this), std::move(name), [s = name_view.c_str(), size] {
            auto res = boost::interprocess::shared_memory_object(boost::interprocess::create_only, s, boost::interprocess::read_write);
            res.truncate(size);
            return handle(res, boost::interprocess::read_write);
        })->second;

        return map(h);
    }

    span_b manager::open_or_create(string name, size_t size) {
        emu::cstring_view name_view = name;
        auto& h = find_or_emplace(static_cast<base&>(*this), std::move(name), [s = name_view.c_str(), size] {
            auto res = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, s, boost::interprocess::read_write);
            res.truncate(size);
            return handle(res, boost::interprocess::read_write);
        })->second;

        return map(h);
    }

    span_b manager::at(emu::cstring_view name) {
        auto it = base::find(name);
        if (it == base::end())
            throw std::runtime_error(fmt::format("No region named {} exists.", name));
        auto& h = it->second;
        return map(h);
    }

    optional<shm_handle> manager::find_handle(const byte* ptr) {
        for (auto& [name, h] : static_cast<base&>(*this)) {

            auto addr = reinterpret_cast<const byte*>(h.get_address());
            auto size = h.get_size();

            if (addr <= ptr && ptr < addr + size) {
                return shm_handle{name, static_cast<size_t>(ptr - addr)};
            }
        }
        return emu::nullopt;
    }


} // namespace sardine::region::host
