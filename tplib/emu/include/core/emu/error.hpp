#pragma once

#include <emu/fwd.hpp>
#include <emu/expected.hpp>

#include <stdexcept>
#include <system_error>

namespace emu
{

    struct error_category: public std::error_category
    {
        const char * name() const noexcept;
        std::string message( int ev ) const;

        static const std::error_category& instance();
    };

    enum class error
    {
        success = 0,

        dlpack_rank_mismatch,
        dlpack_type_mismatch,
        dlpack_strides_not_supported,
        dlpack_read_only,
        dlpack_unkown_device_type,
        dlpack_unkown_data_type_code,

        pointer_device_not_found,
        pointer_maps_file_not_found,
    };

    /**
     * @brief Return a std::error_code from a emu::error
     *
     * @param e The emu::error
     * @return The std::error_code
     */
    std::error_code make_error_code( error e );

    // special case to be used with the error macro below to work with std::error_code.
    inline std::error_code make_error_code( std::error_code e ) { return e; }

    /**
     * @brief Return a unexpected<std::error_code> from a multiple types of error
     *
     * This allow to create result in an error state without knowing the value_type.
     *
     * @param e The emu::error
     * @return The std::error_code
     */
    unexpected<std::error_code> make_unexpected( std::error_code e );
    unexpected<std::error_code> make_unexpected( error e );
    unexpected<std::error_code> make_unexpected( std::errc e );

    [[noreturn]] void throw_error( std::error_code e );
    [[noreturn]] void throw_error( error e );
    [[noreturn]] void throw_error( std::errc e );

    /**
     * @brief This is a helper type that is used to bypass the optional/expected limitation of not being able to hold a reference.
     *
     * @tparam T
     */
    template<typename T>
    using return_type = std::conditional_t<is_ref<T>, std::reference_wrapper<rm_ref<T>>, T>;

    template<typename T>
    using result = expected<return_type<T>, std::error_code>;

namespace detail
{

    struct pretty_error_code
    {
        std::error_code ec;
    };

} // namespace detail

    inline detail::pretty_error_code pretty(std::error_code ec) noexcept {
        return {ec};
    }

} // namespace emu

// ########################
// # error handling macro #
// ########################

/// emu provides a set of macro that use the standard error code system and the standard system_error exception.
/// The first keyword will determine the behavior when there is no error. *_TRUE_* macro will do nothing and continue
/// execution. *_UNWRAP_* will dereference for assignment or return.
/// The second keyword will determine what to do with the error. It can be either *_RETURN_UN_EC_* which stand for
/// "return unexpected error code". It relies on ADL to find the `make_error_code` function in the same
/// namespace than the errc argument.

#define EMU_TRUE_OR_RETURN_UN_EC(expr, errc)                      \
    do {                                                          \
        using ::emu::make_error_code;                             \
        if (EMU_UNLIKELY( not (expr) ))                           \
            return ::emu::make_unexpected(make_error_code(errc)); \
    } while (false)

#define EMU_TRUE_OR_RETURN_UN_EC_LOG(expr, errc, ...)             \
    do {                                                          \
        using ::emu::make_error_code;                             \
        if (EMU_UNLIKELY(not(expr))) {                            \
            EMU_COLD_LOGGER(__VA_ARGS__);                         \
            return ::emu::make_unexpected(make_error_code(errc)); \
        }                                                         \
    } while (false)

#define EMU_UNWRAP_OR_RETURN_UN_EC(expr, errc)         \
    ({                                                 \
        auto&& value__ = (maybe);                      \
        EMU_TRUE_OR_RETURN_UN_EC(value__, errc);       \
        *EMU_FWD(value__);                             \
    })

#define EMU_UNWRAP_OR_RETURN_UN_EC_LOG(expr, errc, ...)                 \
    ({                                                                  \
        auto&& value__ = (expr);                                        \
        EMU_TRUE_OR_RETURN_UN_EC_LOG(value__, errc, __VA_ARGS__);       \
        *EMU_FWD(value__);                                              \
    })

#define EMU_TRUE_OR_THROW_ERROR(expr, errc)            \
    do {                                               \
        using ::emu::make_error_code;                  \
        if (EMU_UNLIKELY( not (expr) ))                \
            ::emu::throw_error(make_error_code(errc)); \
    } while (false)

#define EMU_TRUE_OR_THROW_ERROR_LOG(expr, errc, ...)            \
    do {                                                        \
        using ::emu::make_error_code;                           \
        if (EMU_UNLIKELY( not (expr) )){                        \
            EMU_COLD_LOGGER(__VA_ARGS__);                       \
            ::emu::throw_error(make_error_code(errc));          \
        }                                                       \
    } while (false)


#define EMU_UNWRAP_OR_THROW(result)                                     \
    ({                                                                  \
        auto&& value__ = (result);                                      \
        EMU_TRUE_OR_THROW_ERROR(value__, ::std::move(value__).error()); \
        *EMU_FWD(value__);                                              \
    })

#define EMU_UNWRAP_OR_THROW_LOG(result, ...)                                             \
    ({                                                                                   \
        auto&& value__ = (result);                                                       \
        EMU_TRUE_OR_THROW_ERROR_LOG(value__, ::std::move(value__).error(), __VA_ARGS__); \
        *EMU_FWD(value__);                                                               \
    })

#define EMU_UNWRAP_OR_THROW_ERROR(maybe, errc)        \
    ({                                                \
        auto&& value__ = (maybe);                     \
        EMU_TRUE_OR_THROW_ERROR(value__, errc);       \
        *EMU_FWD(value__);                            \
    })

#define EMU_UNWRAP_OR_THROW_ERROR_LOG(maybe, errc, ...)                \
    ({                                                                 \
        auto&& value__ = (maybe);                                      \
        EMU_TRUE_OR_THROW_ERROR_LOG(value__, errc, __VA_ARGS__);       \
        *EMU_FWD(value__);                                             \
    })

template <>
struct std::is_error_code_enum< emu::error > : std::true_type {};

template<typename Char>
struct fmt::formatter<emu::detail::pretty_error_code, Char>
{
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const emu::detail::pretty_error_code& error, format_context& ctx) const -> format_context::iterator {
        return fmt::format_to(ctx.out(), "{}/{}: {}",
            error.ec.category().name(),
            error.ec.value(),
            error.ec.message()
        );
    }

};
