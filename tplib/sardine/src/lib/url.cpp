#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <sardine/url.hpp>

#include <sardine/type.hpp>
#include <sardine/region/local.hpp>
#include <sardine/region/host.hpp>
#include <sardine/region/managed.hpp>
#include <sardine/memory_converter.hpp>
#include <sardine/region_register.hpp>

#include <emu/pointer.hpp>

namespace sardine::detail
{

    inline auto byte_converter_for(emu::dlpack::device_type_t requested_dt) {
        return [requested_dt]( bytes_and_device res ) -> result< container_b > {

            if ( requested_dt == res.device.device_type // The device type match
              or requested_dt == emu::dlpack::device_type_t::kDLExtDev) // the requested device type is unspecified
                return container_b(res.data, std::move(res.capsule));

            // steal the capsule from the bytes_and_device.
            auto cap = std::move(res.capsule);

            return registry::convert_bytes(std::move(res), requested_dt);
        };
    }

    // TODO: flatten the optional of result.
    // I think it's because we could try other regions if detail::bytes_from_url fails.
    result<container_b> bytes_from_url( const url_view& u, emu::dlpack::device_type_t requested_dt) {
        auto scheme = u.scheme();

        result<bytes_and_device> bytes_and_device;

        if      ( scheme == region::host::url_scheme )
            bytes_and_device = region::host::bytes_from_url( u );

        else if ( scheme == region::managed::url_scheme )
            bytes_and_device = region::managed::bytes_from_url( u );

        else if ( scheme == local::url_scheme )
            bytes_and_device = local::bytes_from_url( u );

        else // Will try all the dynamically registered regions.
            bytes_and_device = registry::url_to_bytes( u );

        return bytes_and_device.and_then(byte_converter_for(requested_dt));
    }

    result<url> url_from_bytes( span_cb bytes, bool allow_local ) {

        if (auto maybe_bytes = registry::revert_convert_bytes(bytes); maybe_bytes)
            bytes = *maybe_bytes; // update the bytes from the reverted bytes and revert the convert_bytes function in bytes_from_url.

        //Could be handle by region::host but it's a bit cleaner to let managed handle itself.
        EMU_UNWRAP_RETURN_IF_TRUE(region::managed::url_from_bytes(bytes));

        EMU_UNWRAP_RETURN_IF_TRUE(region::host::url_from_bytes(bytes));

        // EMU_UNWRAP_RETURN_IF_TRUE(region::cuda::device::url_from_bytes(bytes));

        EMU_UNWRAP_RETURN_IF_TRUE(registry::url_from_bytes(bytes));

        // Should be dead last, because it will always return a url.
        if (allow_local)
            EMU_UNWRAP_RETURN_IF_TRUE(local::url_from_bytes(bytes));



        return make_unexpected( error::url_resource_not_registered );
    }

} // namespace sardine::detail
