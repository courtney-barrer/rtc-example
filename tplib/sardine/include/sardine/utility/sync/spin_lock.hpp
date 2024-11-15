#pragma once

#include <sardine/config.hpp>

#include <boost/atomic/ipc_atomic.hpp>

namespace sardine::sync
{
    enum class LockState {Locked, Unlocked};

    struct SpinLock {
        boost::ipc_atomic<LockState> state = LockState::Locked;

        SpinLock() = default;

        SpinLock(LockState state) : state(state) {}

        void lock()
        {
            while (state.exchange(LockState::Locked, boost::memory_order_acquire) == LockState::Locked) {
            /* busy-wait */
            }
        }

        bool try_lock()
        {
            return state.exchange(LockState::Locked, boost::memory_order_acquire) != LockState::Locked;
        }

        void unlock()
        {
            state.store(LockState::Unlocked, boost::memory_order_release);
        }

        bool is_locked() const
        {
            return state.load(boost::memory_order_acquire) == LockState::Locked;
        }
    };

} // namespace sardine::sync
