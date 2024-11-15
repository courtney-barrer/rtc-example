#include <baldr/utility/component_info.hpp>

#include <emu/associative_container.hpp>

#include <fmt/core.h>

#include <string>

namespace baldr
{
    ComponentInfo& ComponentInfoManager::create_component_info(std::string_view name) {
        namespace fs = std::filesystem;

        sardine::sync::scoped_lock lock(mut);

        // Define a directory for the temporary file (e.g., /tmp)
        fs::path temp_dir = fs::temp_directory_path() / component_info_resource_name;

        // create the directory if it does not exist
        fs::create_directories(temp_dir);

        // Create a unique temporary file name with a prefix
        fs::path temp_file = temp_dir / name;

        // create the temporary file
        std::ofstream(temp_file.c_str());

        auto& res = components.emplace_back(name, temp_file.c_str(), components.get_allocator());

        return res;
    }

    emu::dict<std::string, boost::interprocess::file_lock>& get_locks() {
        static emu::dict<std::string, boost::interprocess::file_lock> locks;
        return locks;
    }

    void acquire_lock(emu::cstring_view name) {
        auto& locks = get_locks();

        fmt::print("Creating lock for {}\n", name);

        auto& file = locks.emplace(name, name.c_str()).first->second;

        fmt::print("Acquiring lock for {}\n", name);
        file.lock();
    }

    void release_lock(std::string_view name) {
        auto& locks = get_locks();

        auto it = locks.find(name);
        if (it != locks.end()) {
            it->second.unlock();
            fmt::print("Releasing lock for {}\n", name);
            locks.erase(it);
        }
    }


} // namespace baldr
