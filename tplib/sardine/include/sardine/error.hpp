#pragma once

#include <sardine/logger.hpp>

#include <emu/error.hpp>

namespace sardine
{
    using emu::result;

    struct error_category: public std::error_category
    {
        [[nodiscard]] std::string message( int ev ) const override;
        [[nodiscard]] const char * name() const noexcept override;

        static const std::error_category& instance();
    };

    enum class error
    {
        success = 0,

        converter_not_found,

        local_url_invalid_host,
        local_url_invalid_path,

        json_parse_reference,
        json_no_conversion,
        json_invalid_json,

        url_resource_not_registered,
        url_unknown_scheme,
        url_param_not_found,

        mapper_rank_mismatch,
        mapper_not_scalar,
        mapper_not_range,
        mapper_not_contiguous,
        mapper_missing_item_size,
        mapper_item_size_mismatch,
        mapper_incompatible_stride,
        mapper_const,

        host_type_not_supported,
        host_url_invalid_path,
        host_url_offset_overflow,
        host_url_size_overflow,
        host_incompatible_shape,

        cuda_module_not_build,
        cuda_invalid_memory_type,
        cuda_type_not_supported,
        cuda_url_invalid_path,
        cuda_url_offset_overflow,
        cuda_url_size_overflow,

        managed_invalid_url_segment_count,

        embedded_type_not_handled,
        embedded_url_missing_json,

        ring_missing_size,
        ring_url_missing_size,
        ring_url_missing_data,
        ring_url_missing_index,
        ring_url_missing_buffer_nb,
        ring_url_missing_policy,
        ring_url_invalid_policy,

        location_cuda_unsupported_source_memory,
        location_cuda_device_region_not_registered,
        location_conversion_not_handle,
        location_could_get_device_pointer,

        python_type_not_supported
    };

    /**
     * @brief Return a std::error_code from a sardine::error
     *
     * @param e
     * @return std::error_code
     */
    std::error_code make_error_code( error e );

    /**
     * @brief Utility function to easily exit from a function that return a expected type.
     *
     * @param e
     * @return tl::unexpected< std::error_code >
     */
    emu::unexpected< std::error_code > make_unexpected( error e );

} // namespace sardine

template<>
struct std::is_error_code_enum< ::sardine::error > : std::true_type { };
