#include <sardine/error.hpp>

namespace sardine
{

    std::error_category const& sardine_category() {
        static const error_category instance;
        return instance;
    }

    const char * error_category::name() const noexcept {
        return "sardine";
    }

    const std::error_category& error_category::instance() {
        static const error_category instance;
        return instance;
    }

    std::string error_category::message( int ev ) const {
        switch (static_cast<error>(ev)) {
            case error::success: return "success";

            case error::converter_not_found: return "no converter found for device type";

            case error::local_url_invalid_host: return "local url invalid host";
            case error::local_url_invalid_path: return "local url invalid path";

            case error::json_parse_reference: return "json parse reference";
            case error::json_no_conversion: return "type is not parsable from json";
            case error::json_invalid_json: return "could not parse json into requested type";

            case error::url_resource_not_registered: return "resource not registered, cannot generate url";
            case error::url_unknown_scheme:          return "url unknown scheme";
            case error::url_param_not_found:        return "url does not have the requested parameter";

            case error::mapper_rank_mismatch:  return "mapper rank mismatch";
            case error::mapper_not_scalar:     return "mapper not scalar";
            case error::mapper_not_range:      return "mapper not range";
            case error::mapper_not_contiguous: return "mapper not contiguous";
            case error::mapper_missing_item_size: return "mapper missing item size";
            case error::mapper_item_size_mismatch: return "mapper item size mismatch";
            case error::mapper_incompatible_stride: return "mapper does not support provided stride";
            case error::mapper_const: return "mapper is const";

            case error::host_type_not_supported:  return "host type not supported";
            case error::host_url_invalid_path:    return "host url invalid path";
            case error::host_url_offset_overflow: return "host url offset overflow";
            case error::host_url_size_overflow:   return "host url size overflow";
            case error::host_incompatible_shape:   return "host opening with a shape that is incompatible with region size";

            case error::managed_invalid_url_segment_count: return "managed invalid url segment count, expected 2";

            case error::embedded_type_not_handled: return "the requested type can not be constructed from json";
            case error::embedded_url_missing_json: return "embedded url is missing the json parameter";

            case error::cuda_module_not_build:  return "sardine cuda module is not build. Recompile with SARDINE_CUDA";
            case error::cuda_invalid_memory_type:  return "cuda invalid memory type";
            case error::cuda_type_not_supported:  return "cuda type not supported";
            case error::cuda_url_invalid_path:    return "cuda url invalid path";
            case error::cuda_url_offset_overflow: return "cuda url offset overflow";
            case error::cuda_url_size_overflow:   return "cuda url size overflow";

            case error::ring_missing_size:          return "ring missing size";
            case error::ring_url_missing_size:      return "ring url missing size";
            case error::ring_url_missing_data:      return "ring url missing data";
            case error::ring_url_missing_index:     return "ring url missing index";
            case error::ring_url_missing_buffer_nb: return "ring url missing buffer nb";
            case error::ring_url_missing_policy:    return "ring url missing policy";
            case error::ring_url_invalid_policy:    return "ring url invalid policy";

            case error::location_cuda_unsupported_source_memory: return "location cuda unsupported source memory";
            case error::location_cuda_device_region_not_registered: return "location cuda device region not registered";
            case error::location_conversion_not_handle: return "no conversion know to the requested device type";
            case error::python_type_not_supported: return "python type is not supported by url";
            case error::location_could_get_device_pointer: return "Could not get the device pointer of the registered region";
        }
        return "unknown";
    }

    std::error_code make_error_code( error e ) {
        return { static_cast<int>(e), error_category::instance() };
    }

    emu::unexpected<std::error_code> make_unexpected( error e ) {
        return emu::unexpected<std::error_code>( make_error_code(e) );
    }

} // namespace sardine
