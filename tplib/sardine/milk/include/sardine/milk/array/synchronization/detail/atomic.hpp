#ifndef OCTOPUS_SYNC_DETAIL_ATOMIC_H
#define OCTOPUS_SYNC_DETAIL_ATOMIC_H

#include <emu/macro.h>

#include <octopus/detail/type.h>

namespace octopus
{

namespace sync
{

    #if not EMU_CUDACC

    /// Define double long long struct as CUDA ulonglong2 for host compiler with same properties.
    struct alignas(16) u128 {
        u64 x, y;

        EMU_HODE u128() = default;
        EMU_HODE u128(const volatile u128 & o) : x(o.x), y(o.y) {}
        EMU_HODE u128& operator=(const volatile u128 & o) noexcept {
            x = o.x; y = o.y;
            return *this;
        }
    };

#else // EMU_CUDACC

    using u128 = ulonglong2;

#endif

    ///
    constexpr EMU_HODE bool operator==(const u128 & lhs, const u128 & rhs) noexcept {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }

    template<typename T>
    struct atomic_ref_t {
        T * ptr;

        EMU_HODE atomic_ref_t(T & data) : ptr(&data) {}

        EMU_HODE void wait(T old) const noexcept {
            T v;
            do{
                v = *ptr;
            }while(v == old);
        }

        EMU_HODE void store(T desired) const noexcept {
            *ptr = desired;
        }

        EMU_HODE T load() const noexcept {
            return *ptr;
        }
    };

namespace atomic
{

    template<typename T>
    EMU_HODE atomic_ref_t<T> create(T & data) {
        return {data};
    }

} // namespace atomic

} // namespace sync

} // namespace octopus

#endif //OCTOPUS_SYNC_DETAIL_ATOMIC_H