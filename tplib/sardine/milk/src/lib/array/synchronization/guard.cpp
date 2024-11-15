#include <cacao/synchronization/guard.hpp>

namespace cacao
{

namespace sync
{

namespace guard
{

namespace detail
{

    /// Links size and alignment requirement with a POD type.
    /// Notes : for now, alignment is ignored since each cacao type have equal size and alignment.
    template<std::size_t TypeSize, std::size_t TypeAlign> struct maching_type;

    template<std::size_t TypeAlign> struct maching_type< 1, TypeAlign> { using type =   u8; };
    template<std::size_t TypeAlign> struct maching_type< 2, TypeAlign> { using type =  u16; };
    template<std::size_t TypeAlign> struct maching_type< 4, TypeAlign> { using type =  u32; };
    template<std::size_t TypeAlign> struct maching_type< 8, TypeAlign> { using type =  u64; };
    template<std::size_t TypeAlign> struct maching_type<16, TypeAlign> { using type = u128; };

    template<std::size_t TypeSize, std::size_t TypeAlign>
    using maching_type_t = typename maching_type<TypeSize, TypeAlign>::type;

    /// Generate guard ptr, lock & unlock value.
    template<std::size_t TypeSize, std::size_t TypeAlign, typename Fn>
    auto call_size_align(Fn && fn, type_t type, byte* ptr) {
        static_assert(TypeSize == TypeAlign, "size and alignment must be equal");

        using DataType = maching_type_t<TypeSize, TypeAlign>;

        DataType
            *data   =  reinterpret_cast<      DataType*>(ptr),
             lock   = *reinterpret_cast<const DataType*>(sync::lock(type)),
             unlock = *reinterpret_cast<const DataType*>(sync::unlock(type));

        return EMU_FWD(fn)(data, lock, unlock);
    }

    /// Check the alignment requirement of the guard.
    template<std::size_t TypeSize, typename Fn>
    auto call_size(Fn && fn, type_t type, byte* ptr) {
        switch (align_of(type))
        {
            case  1: return call_size_align<TypeSize,  1>(EMU_FWD(fn), type, ptr); break;
            case  2: return call_size_align<TypeSize,  2>(EMU_FWD(fn), type, ptr); break;
            case  4: return call_size_align<TypeSize,  4>(EMU_FWD(fn), type, ptr); break;
            case  8: return call_size_align<TypeSize,  8>(EMU_FWD(fn), type, ptr); break;
            case 16: return call_size_align<TypeSize, 16>(EMU_FWD(fn), type, ptr); break;
        }
        EMU_ASSUME_UNREACHABLE_MSG("Call with invalid alignment");
    }

    /// Check the size requirement of the guard.
    template<typename Fn>
    auto call(Fn && fn, type_t type, byte* ptr) {
        switch (size(type))
        {
            case  1: return call_size< 1>(EMU_FWD(fn), type, ptr); break;
            case  2: return call_size< 2>(EMU_FWD(fn), type, ptr); break;
            case  4: return call_size< 4>(EMU_FWD(fn), type, ptr); break;
            case  8: return call_size< 8>(EMU_FWD(fn), type, ptr); break;
            case 16: return call_size<16>(EMU_FWD(fn), type, ptr); break;
        }
        EMU_ASSUME_UNREACHABLE_MSG("Call with invalid size");
    }

} // namespace detail

    void wait_and_reset(byte* ptr, type_t type, std::size_t guard_nb)
    {
        detail::call([guard_nb](auto* ptr, auto lock, auto unlock){
            wait_equal(*ptr, unlock);
            memcopy(ptr, &lock, guard_nb);
        }, type, ptr);
    }

    void set(byte* ptr, type_t type, std::size_t guard_nb, GuardMode mode)
    {
        detail::call([guard_nb, mode](auto* ptr, auto lock, auto unlock){
            if (mode == GuardMode::lock)
                memcopy(ptr, &lock, guard_nb);
            else
                memcopy(ptr, &unlock, guard_nb);

        }, type, ptr);
    }

    /// Perform active wait on cpu memory on each guards and reset them before returning.
    void wait_and_reset(Array & array, SyncData s_data)
    {
    }

    /// Set guards to the desired state.
    void set(Array & array, SyncData s_data, GuardMode mode)
    {

    }

#if EMU_CUDA

    void wait_and_reset(byte* ptr, type_t type, std::size_t guard_nb, stream_cref_t stream)
    {

    }

    void set(byte* ptr, type_t type, std::size_t guard_nb, GuardMode mode, stream_cref_t stream)
    {

    }

    /// Perform active wait on CUDA memory on each guards and reset them before returning.
    /// Memory and stream must be on se same location.
    void wait_and_reset(Array & array, SyncData s_data, stream_cref_t stream)
    {

    }

    /// Set guards to the desired state.
    void set(Array & array, SyncData s_data, GuardMode mode, stream_cref_t stream)
    {

    }

#endif

} // namespace guard

} // namespace sync

} // namespace cacao
