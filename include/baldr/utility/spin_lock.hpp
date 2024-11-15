#pragma once

#include <baldr/config.hpp>
#include <boost/atomic/ipc_atomic.hpp>

namespace baldr
{
    enum class LockState {Locked, Unlocked};

    struct SpinLock {
        boost::ipc_atomic<LockState> state[6];

        SpinLock() = default;

        SpinLock(LockState state) : state(state) {}

        void lock(size_t idx)
        {
            while (state[idx].exchange(LockState::Locked, boost::memory_order_acquire) == LockState::Locked) {
            /* busy-wait */
            }
        }

        bool try_lock(size_t idx)
        {
            return state[idx].exchange(LockState::Locked, boost::memory_order_acquire) != LockState::Locked;
        }

        void unlock()
        {
            for( int i = 0; i < 6; i++)
                state[i].store(LockState::Unlocked, boost::memory_order_release);
        }
    };

} // namespace baldr
