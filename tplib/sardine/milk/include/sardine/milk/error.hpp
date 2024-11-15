#pragma once

#include <emu/error.hpp>

extern "C"
{
    #include <ImageStreamIO/ImageStreamIOError.h>
} // extern "C"

namespace sardine::milk
{

    using emu::result;

    struct error_category: public std::error_category
    {
        [[nodiscard]] std::string message( int ev ) const override;
        [[nodiscard]] const char * name() const noexcept override;

        static const std::error_category& instance();
    };

    enum class errc
    {
        success = IMAGESTREAMIO_SUCCESS, // reusing ImageStreamIO error codes

        milk_generic_error = IMAGESTREAMIO_FAILURE,
        milk_invalid_argument = IMAGESTREAMIO_INVALIDARG,
        milk_not_implemented = IMAGESTREAMIO_NOTIMPL,
        milk_bad_allocation = IMAGESTREAMIO_BADALLOC,
        milk_file_open = IMAGESTREAMIO_FILEOPEN,
        milk_file_seek = IMAGESTREAMIO_FILESEEK,
        milk_file_write = IMAGESTREAMIO_FILEWRITE,
        milk_file_exists = IMAGESTREAMIO_FILEEXISTS,
        milk_inode = IMAGESTREAMIO_INODE,
        milk_mmap = IMAGESTREAMIO_MMAP,
        milk_semaphore_init = IMAGESTREAMIO_SEMINIT,
        milk_version = IMAGESTREAMIO_VERSION,

        array_open_type_mismatch,
        array_open_shape_mismatch,
        array_open_location_mismatch,
        array_data_type_not_handled,

        fps_field_does_not_exist,
        fps_field_already_exist,
        fps_field_type_not_handled,
        fps_field_type_mismatch,
    };

    /**
     * @brief Return a std::error_code from a sardine::error
     *
     * @param e
     * @return std::error_code
     */
    std::error_code make_error_code( errc e );

    /**
     * @brief Utility function to easily exit from a function that return a expected type.
     *
     * @param e
     * @return tl::unexpected< std::error_code >
     */
    emu::unexpected< std::error_code > make_unexpected( errc e );

} // namespace sardine::milk

#define MILK_CHECK_RETURN_UN_EC(expr) \
    EMU_SUCCESS_OR_RETURN_UN_EC(static_cast<::sardine::milk::errc>(expr), ::sardine::milk::errc::success)

#define MILK_CHECK_RETURN_UN_EC_LOG(expr, ...) \
    EMU_SUCCESS_OR_RETURN_UN_EC_LOG(static_cast<::sardine::milk::errc>(expr), ::sardine::milk::errc::success, __VA_ARGS__)

#define MILK_CHECK_THROW(expr) \
    EMU_SUCCESS_OR_THROW(static_cast<::sardine::milk::errc>(expr), ::sardine::milk::errc::success)

#define MILK_CHECK_THROW_LOG(expr, ...) \
EMU_SUCCESS_OR_THROW_LOG(static_cast<::sardine::milk::errc>(expr), ::sardine::milk::errc::success, __VA_ARGS__)


template<>
struct std::is_error_code_enum< ::sardine::milk::errc > : std::true_type { };
