#include <sardine/milk/error.hpp>

namespace sardine::milk
{

    std::string error_category::message( int ev ) const {
        switch (static_cast<errc>(ev)) {
            case errc::success:
                return "success";

            case errc::milk_generic_error:
                return "generic error";
            case errc::milk_invalid_argument:
                return "invalid argument";
            case errc::milk_not_implemented:
                return "not implemented";
            case errc::milk_bad_allocation:
                return "bad allocation";
            case errc::milk_file_open:
                return "file open";
            case errc::milk_file_seek:
                return "file seek";
            case errc::milk_file_write:
                return "file write";
            case errc::milk_file_exists:
                return "file exists";
            case errc::milk_inode:
                return "cannot obtain inode";
            case errc::milk_mmap:
                return "could not map file";
            case errc::milk_semaphore_init:
                return "semaphore init";
            case errc::milk_version:
                return "incompatible version";

            case errc::array_open_type_mismatch:
                return "array open type mismatch";
            case errc::array_open_shape_mismatch:
                return "array open shape mismatch";
            case errc::array_open_location_mismatch:
                return "array open location mismatch";
            case errc::array_data_type_not_handled:
                return "array data type not handled";

            case errc::fps_field_does_not_exist:
                return "fps field does not exist";
            case errc::fps_field_already_exist:
                return "fps field already exist";
            case errc::fps_field_type_not_handled:
                return "fps field type not handled";
            case errc::fps_field_type_mismatch:
                return "fps field type mismatch";
        }
        return "unknown";
    }
    const char * error_category::name() const noexcept {
        return "milk";
    }

    const std::error_category& error_category::instance() {
        static const error_category instance;
        return instance;
    }

    std::error_code make_error_code( errc e ) {
        return { static_cast<int>(e), error_category::instance() };
    }

    emu::unexpected<std::error_code> make_unexpected( errc e ) {
        return emu::unexpected<std::error_code>( make_error_code(e) );
    }

} // namespace sardine::milk
