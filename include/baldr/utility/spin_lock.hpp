#pragma once

#include <boost/atomic/ipc_atomic.hpp>

namespace baldr
{
    enum class LockState {Locked, Unlocked};

    struct SpinLock {
        myboost::ipc_atomic<LockState> state[6];

        SpinLock() = default;

        SpinLock(LockState state) : state(state) {}

        void lock(size_t idx)
        {
            while (state[idx].exchange(LockState::Locked, myboost::memory_order_acquire) == LockState::Locked) {
            /* busy-wait */
            }
        }

        bool try_lock(size_t idx)
        {
            return state[idx].exchange(LockState::Locked, myboost::memory_order_acquire) != LockState::Locked;
        }

        void unlock()
        {
            for( int i = 0; i < 6; i++)
                state[i].store(LockState::Unlocked, myboost::memory_order_release);
        }
    };

} // namespace baldr
