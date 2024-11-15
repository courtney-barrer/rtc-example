#ifndef CACAO_SYNC_GUARD_H
#define CACAO_SYNC_GUARD_H

#include <cacao/sync/type.h>
#include <cacao/sync/detail/lock.h>
#include <cacao/sync/detail/atomic.h>

#include <emu/macro.h>
#include <emu/assert.h>

namespace cacao
{

namespace sync
{

namespace guard
{

    void wait_and_reset(byte* ptr, type_t type, std::size_t guard_nb);

    void set(byte* ptr, type_t type, std::size_t guard_nb, GuardMode mode);

    /// Perform active wait on cpu memory on each guards and reset them before returning.
    void wait_and_reset(Array & array, SyncData s_data);

    /// Set guards to the desired state.
    void set(Array & array, SyncData s_data, GuardMode mode);

#if EMU_CUDA

    void wait_and_reset(byte* ptr, type_t type, std::size_t guard_nb, stream_cref_t stream);

    void set(byte* ptr, type_t type, std::size_t guard_nb, GuardMode mode, stream_cref_t stream);

    /// Perform active wait on CUDA memory on each guards and reset them before returning.
    /// Memory and stream must be on se same location.
    void wait_and_reset(Array & array, SyncData s_data, stream_cref_t stream);

    /// Set guards to the desired state.
    void set(Array & array, SyncData s_data, GuardMode mode, stream_cref_t stream);

#endif

} // namespace guard

// namespace stream
// {

//     #if EMU_CUDA

//     /// Perform active wait on CUDA memory on each guards and reset them before returning.
//     /// Memory and stream must be on se same location.
//     void wait_and_reset(Array & array, SyncData s_data, stream_cref_t stream);

//     /// Set guards to the desired state.
//     void set(Array & array, SyncData s_data, GuardMode mode, stream_cref_t stream);

// #endif

// } // namespace stream

} // namespace sync

} // namespace cacao

#endif //CACAO_SYNC_GUARD_H