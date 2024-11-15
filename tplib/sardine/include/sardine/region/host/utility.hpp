#pragma once

#include <sardine/type.hpp>

#include <emu/cstring_view.hpp>

#include <boost/interprocess/mapped_region.hpp>

namespace sardine::region::host
{
    // A handle to a shared memory region.
    using handle = boost::interprocess::mapped_region;

    // A pointer to a shared memory location.
    struct shm_handle {
        cstring_view name;
        size_t offset;
    };

    inline auto map(const handle &r) -> span_b {
        return {static_cast<byte*>(r.get_address()), r.get_size()};
    }

} // namespace sardine::region::host
