#pragma once

#include <memory>
#include <sardine/config.hpp>
#include <sardine/region/managed.hpp>
#include <sardine/utility/sync/spin_lock.hpp>

#include <boost/atomic/ipc_atomic.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <fmt/core.h>

namespace sardine::sync
{

    namespace bi = boost::interprocess;

    using scoped_lock = bi::scoped_lock<bi::interprocess_mutex>;

    using lock_vector = region::managed::stable_vector<boost::ipc_atomic<LockState>>;
    using lock_iterator = lock_vector::iterator;

    struct Barrier;

    struct Waiter {
        Barrier* barrier;
        lock_iterator state;

        Waiter(Barrier* barrier);

        Waiter(Waiter&& other)
            : barrier(std::exchange(other.barrier, nullptr))
            , state(other.state)
        {}

        ~Waiter();

        void wait();
    };

    struct Notifier {
        Barrier* barrier;

        Notifier(Barrier* barrier);

        Notifier(Notifier&& other)
            : barrier(std::exchange(other.barrier, nullptr))
        {}

        ~Notifier();

        void notify();
    };

    struct Barrier {
        bi::interprocess_mutex     mutex;
        bi::interprocess_condition cond;

        boost::ipc_atomic<size_t> count;
        boost::ipc_atomic<size_t> total_count;
        lock_vector states;

        Barrier(region::managed::allocator<LockState> alloc)
            : count(0), total_count(0), states(alloc)
        {
            if (not boost::ipc_atomic<LockState>::always_has_native_wait_notify)
                throw std::runtime_error("Atomic type does not support native wait/notify");
        }

        Waiter   get_waiter  () { return Waiter(this); }
        Notifier get_notifier() { return Notifier(this); }

        void notify_all() {
            scoped_lock lock(mutex);
            cond.notify_all();
        }

    };

    inline void Waiter::wait() {

        while (state->exchange(LockState::Locked, boost::memory_order_acquire) == LockState::Locked) {
            scoped_lock lock(barrier->mutex);

            barrier->cond.wait(lock);
        }

    }

    inline void Notifier::notify() {
        scoped_lock lock(barrier->mutex);

        // Check if we are the last one to notify
        if (barrier->count.sub(1) == 0) {
            // Unlock all waiters
            for (auto& state : barrier->states)
                state.store(LockState::Unlocked, boost::memory_order_release);
            // And notify them
            barrier->cond.notify_all();

            // Finally reset the count
            barrier->count.store(barrier->total_count.load());
        }
    }

} // namespace sardine::sync

template<typename Alloc>
struct std::uses_allocator<sardine::sync::Barrier, Alloc> : std::true_type {};
