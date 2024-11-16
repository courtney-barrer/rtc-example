#include <sardine/sink.hpp>
#include <sardine/logger.hpp>

#include <unordered_set>

namespace sardine
{

    using capsule_holder = std::unordered_set<emu::capsule>;

    capsule_holder& capsule_holder_instance()
    {
        static capsule_holder holder;
        return holder;
    }

    void sink(emu::capsule&& c) {
        // Only insert if the capsule is not empty.
        if (c) {
#ifndef NDEBUG // Debug mode
            EMU_LOGGER("Sinking capsule\n");
#endif
            capsule_holder_instance().insert(std::move(c));
        }
    }

} // namespace sardine
