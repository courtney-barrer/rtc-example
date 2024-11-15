#ifndef FPS_DETAIL_UTILITY_H
#define FPS_DETAIL_UTILITY_H

#include <fps/detail/type.h>

#include <octopus/detail/type.h>

#include <emu/assert.h>

#include <fmt/core.h>

#include <stdexcept>

namespace fps
{
    constexpr inline fps_int_type_t fps_int_type(type_t type) {
        return static_cast<fps_int_type_t>(type);
    }

    constexpr inline type_t type(fps_int_type_t type) {
        // TODO: Need to check if type is one of the handled types.
        return static_cast<type_t>(type);
    }


    inline constexpr const char * name(type_t type) noexcept {
        switch (type)
        {
            case type_t::i32: return "int32";
            case type_t::i64: return "int64";
            case type_t::f32: return "float";
            case type_t::f64: return "double";
            case type_t::str: return "str";
        }
        EMU_UNREACHABLE;
    }

namespace detail
{
    template<typename T>
    constexpr type_t type_of() noexcept;

    #define FPS_TYPE_OF(T)                                                      \
    template<> constexpr type_t type_of<octopus::T>() noexcept { return type_t::T;  }

    FPS_TYPE_OF(i32)
    FPS_TYPE_OF(i64)
    FPS_TYPE_OF(f32)
    FPS_TYPE_OF(f64)

    // Special case for `std::string`
    template<> constexpr type_t type_of<std::string>() noexcept { return type_t::str; }
    template<> constexpr type_t type_of<char*>() noexcept { return type_t::str; }

    #undef FPS_TYPE_OF

} // namespace detail

    template<typename T>
    constexpr type_t type_of = detail::type_of<T>();

    inline void throw_type_not_handle(type_t type) {
        throw std::runtime_error(fmt::format("Type {} not handle", name(type)));
    }

    template<typename T>
    void throw_if_type_mismatch(type_t type) {
        auto t_type = type_of<T>();
        if (type != t_type)
            throw std::runtime_error(fmt::format("Type mismatch between {} and {}", name(type), name(t_type)));
    }

} // namespace fps

template <> struct fmt::formatter<fps::type_t> {

    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(fps::type_t t, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{}", fps::name(t));
    }
};

#endif //FPS_DETAIL_UTILITY_H