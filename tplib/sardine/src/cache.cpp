#include <sardine/cache.hpp>

#include <fmt/format.h>

#include <list>
#include <filesystem>

namespace sardine::cache
{

namespace detail
{

    using region::managed_t;

    constexpr size_t default_cache_size = 1024 * 1024;


    managed_t get_next_managed_cache(size_t required_space) {
        static size_t current_idx = 0;
        if (required_space > 1024 * 1024 * 1024) {
            throw std::runtime_error("Cache size too large");
        }
        return region::managed::open_or_create(fmt::format("sardine_cache_{}", current_idx++), std::max(required_space, default_cache_size));
    }

    struct cache_manager {
        using cache_list_t = std::list<managed_t>;

    private:
        cache_manager() = default;
    public:
        static cache_manager& instance() {
            static cache_manager instance;
            return instance;
        }

        cache_list_t caches;

        managed_t request_cache(size_t required_space) {
            // Looking into existing caches
            for (auto& cache : caches) {
                if (cache.shm().get_free_memory() >= required_space)
                    return cache;
            }
            // Creating a new cache until we find one with enough space
            while(true) {
                //
                auto cache = caches.emplace_back(get_next_managed_cache(required_space));
                if (cache.shm().get_free_memory() >= required_space)
                    return cache;
            }
        }
    };

    managed_t get_managed_cache(size_t required_space)
    {
        return cache_manager::instance().request_cache(required_space);
    }

} // namespace detail

    void clear()
    {
        // removes all files in /dev/shm that start with "sardine_cache_"
        std::string prefix = "sardine_cache_";
        std::filesystem::path shm_dir = "/dev/shm";

        for (const auto& entry : std::filesystem::directory_iterator(shm_dir))
        {
            if (entry.is_regular_file() && entry.path().filename().string().starts_with(prefix))
            {
                fmt::print("Removing {}\n", entry.path().string());
                std::filesystem::remove(entry.path());
            }
        }
    }

} // namespace sardine::cache
