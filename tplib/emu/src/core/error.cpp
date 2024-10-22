#include <emu/error.hpp>

namespace emu
{

    const char * error_category::name() const noexcept {
        return "emu";
    }

    std::string error_category::message( int ev ) const {
        switch (static_cast<error>(ev)) {
            case error::success: return "success";

            case error::dlpack_rank_mismatch: return "dlpack rank mismatch";
            case error::dlpack_type_mismatch: return "dlpack type mismatch";
            case error::dlpack_strides_not_supported: return "dlpack strides not supported";
            case error::dlpack_read_only: return "dlpack trying to access read only memory";
            case error::dlpack_unkown_device_type: return "dlpack unknow device type";
            case error::dlpack_unkown_data_type_code: return "dlpack unknow data type code";

            case error::pointer_device_not_found: return "pointer's device not found";
            case error::pointer_maps_file_not_found: return "pointer's maps file not found";

        }
        return "unknown";
    }

    const std::error_category& error_category::instance() {
        static const error_category instance;
        return instance;
    }

    std::error_code make_error_code( error e ) {
        return { static_cast<int>(e), error_category::instance() };
    }

    unexpected<std::error_code> make_unexpected( std::error_code e ) {
        return unexpected<std::error_code>{ e };
    }

    unexpected<std::error_code> make_unexpected( error e ) {
        return make_unexpected( make_error_code(e) );
    }

    unexpected<std::error_code> make_unexpected( std::errc e ) {
        return make_unexpected( std::make_error_code(e) );
    }

    void throw_error( std::error_code e ) {
        throw std::system_error( e );
    }

    void throw_error( error e ) {
        throw_error( make_error_code(e) );
    }

    void throw_error( std::errc e ) {
        throw_error( std::make_error_code(e) );
    }

} // namespace emu
