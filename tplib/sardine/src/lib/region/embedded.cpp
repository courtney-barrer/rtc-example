#include <sardine/region/embedded.hpp>

#include <list>

namespace sardine::region::embedded::detail
{
    using capsule_list = std::list<emu::capsule>;

    capsule_list& cache() {
        static capsule_list instance;
        return instance;
    }

    void keep_alive(emu::capsule c) {
        cache().push_back(std::move(c));

    }


} // namespace sardine::region::embedded::detail
