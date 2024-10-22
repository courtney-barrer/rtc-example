#ifndef CACAO_SYNC_TYPE_H
#define CACAO_SYNC_TYPE_H

#include <cstddef>

namespace cacao
{

namespace sync
{

    /// Define how to perform synchronization operations on array.
    enum class Mode {
        internal, ///< Let the shm handle operations.
        guard,    ///< Use active polling on memory.
        stream    ///< Use CUDA operations. Only with CUDA memory.
    };

    /// Define which memory region is gonna bu used for polling for polling and stream synchronizations.
    enum class Position {
        in, ///< Use data memory region.
        out ///< Use meta memory region.
    };

    /// Define all needed informations for polling and stream synchronizations.
    struct SyncData {
        Position position;

        std::size_t offset;   ///< Offset from
        std::size_t guard_nb; ///< Number of guard to check.
    };

    enum class GuardMode {
        lock,  ///< Specify guard lock.
        unlock ///< Specify guard unlock.
    };

} // namespace sync

} // namespace cacao

#endif //CACAO_SYNC_TYPE_H