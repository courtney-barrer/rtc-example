#ifndef CACAO_SYNC_DETAIL_LOCK_H
#define CACAO_SYNC_DETAIL_LOCK_H

#include <octopus/detail/type.h>

#include <emu/macro.h>
#include <emu/type_traits.h>

#include <limits>

namespace cacao
{

namespace sync
{

    /// Returns guard value for integral type such as signed and insigned integer.
    /// Guard is maximum positive value.
    template<typename T, emu::EnableIf<emu::IsIntegral<T>> = true>
    EMU_HODE constexpr T guard_lock(){ return std::numeric_limits<T>::max(); }

    /// Returns guard value for floating point type.
    /// Guard is corresponding NaN value.
    template<typename T, emu::EnableIf<emu::IsFloatingPoint<T>> = true>
    EMU_HODE constexpr T guard_lock(){ return std::numeric_limits<T>::quiet_NaN(); }

    /// Returns a non blocking value for guards.
    template<typename T>
    EMU_HODE constexpr T guard_unlock(){ return T{}; }

namespace detail
{

    template<typename T>
    std::array<byte, sizeof(T)> guard_lock() {
        std::array<byte, sizeof(T)> lock;
        auto tmp = guard_lock<T>();
        emu::copy(reinterpret_cast<const byte*>(&tmp), lock.data(), sizeof(T));
        return lock;
    }

    template<typename T>
    std::array<byte, sizeof(T)> guard_unlock() {
        std::array<byte, sizeof(T)> lock;
        auto tmp = guard_unlock<T>();
        emu::copy(reinterpret_cast<const byte*>(&tmp), lock.data(), sizeof(T));
        return lock;
    }

    template<typename T>
    const byte* static_guard_lock() {
        static std::array<byte, sizeof(T)> lock = guard_lock<T>();
        return lock.data();
    }

    template<typename T>
    const byte* static_guard_unlock() {
        static std::array<byte, sizeof(T)> lock = guard_unlock<T>();
        return lock.data();
    }

} // namespace detail

    inline const byte* lock(type_t type) {
        switch (type)
        {
            case type_t:: u8: return detail::static_guard_lock< u8>();
            case type_t:: i8: return detail::static_guard_lock< i8>();
            case type_t::u16: return detail::static_guard_lock<u16>();
            case type_t::i16: return detail::static_guard_lock<i16>();
            case type_t::u32: return detail::static_guard_lock<u32>();
            case type_t::i32: return detail::static_guard_lock<i32>();
            case type_t::u64: return detail::static_guard_lock<u64>();
            case type_t::i64: return detail::static_guard_lock<i64>();
            case type_t::f16: return detail::static_guard_lock<f16>();
            case type_t::f32: return detail::static_guard_lock<f32>();
            case type_t::f64: return detail::static_guard_lock<f64>();
            case type_t::c16: return detail::static_guard_lock<c16>();
            case type_t::c32: return detail::static_guard_lock<c32>();
            case type_t::c64: return detail::static_guard_lock<c64>();
        }
        EMU_UNREACHABLE;
    }

    inline const byte* unlock(type_t type) {
        switch (type)
        {
            case type_t:: u8: return detail::static_guard_unlock< u8>();
            case type_t:: i8: return detail::static_guard_unlock< i8>();
            case type_t::u16: return detail::static_guard_unlock<u16>();
            case type_t::i16: return detail::static_guard_unlock<i16>();
            case type_t::u32: return detail::static_guard_unlock<u32>();
            case type_t::i32: return detail::static_guard_unlock<i32>();
            case type_t::u64: return detail::static_guard_unlock<u64>();
            case type_t::i64: return detail::static_guard_unlock<i64>();
            case type_t::f16: return detail::static_guard_unlock<f16>();
            case type_t::f32: return detail::static_guard_unlock<f32>();
            case type_t::f64: return detail::static_guard_unlock<f64>();
            case type_t::c16: return detail::static_guard_unlock<c16>();
            case type_t::c32: return detail::static_guard_unlock<c32>();
            case type_t::c64: return detail::static_guard_unlock<c64>();
        }
        EMU_UNREACHABLE;
    }

} // namespace sync

} // namespace cacao

#endif //CACAO_SYNC_DETAIL_LOCK_H