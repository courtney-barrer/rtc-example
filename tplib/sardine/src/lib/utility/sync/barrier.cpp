#include <sardine/utility/sync/barrier.hpp>

namespace sardine::sync
{


    Notifier::Notifier(Barrier* barrier)
        : barrier(barrier)
    {
        barrier->total_count.add(1);
        barrier->count.add(1);
    }

    Notifier::~Notifier() {
        if (barrier) {
            fmt::print("Notifier: deletion\n");
            barrier->total_count.sub(1);
            barrier->count.sub(1);
        }
    }

    Waiter::Waiter(Barrier* barrier)
        : barrier(barrier)
        , state([&]{
            barrier->states.emplace_back(LockState::Locked);

            return barrier->states.end() - 1;
        }())
    {}

    Waiter::~Waiter() {
        if (barrier)
            barrier->states.erase(state);
    }


} // namespace sardine::sync
