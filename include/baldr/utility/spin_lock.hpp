#pragma once

#include <boost/atomic/ipc_atomic.hpp>

namespace baldr
{
    enum class LockState {Locked, Unlocked};

    struct SpinLock {
        myboost::ipc_atomic<LockState> state = LockState::Locked;

        SpinLock() = default;

        SpinLock(LockState state) : state(state) {}

        void lock()
        {
            while (state.exchange(LockState::Locked, myboost::memory_order_acquire) == LockState::Locked) {
            /* busy-wait */
            }
        }

        bool try_lock()
        {
            return state.exchange(LockState::Locked, myboost::memory_order_acquire) != LockState::Locked;
        }

        void unlock()
        {
            state.store(LockState::Unlocked, myboost::memory_order_release);
        }
    };

} // namespace baldr
