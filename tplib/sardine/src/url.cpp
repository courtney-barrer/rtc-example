#include <sardine/url.hpp>

#include <sardine/type.hpp>
#include <sardine/region/local.hpp>
#include <sardine/region/host.hpp>
#include <sardine/region/cuda/device.hpp>
#include <sardine/region/managed.hpp>

#include <emu/pointer.hpp>

namespace sardine::detail
{

    // TODO: flatten the optional of result.
    // I think it's because we could try other regions if detail::bytes_from_url fails.
    result<bytes_and_device> bytes_from_url( const url_view& u ) {
        auto scheme = u.scheme();

        if ( scheme == region::host::url_scheme )
            return region::host::bytes_from_url( u );
        else if ( scheme == region::cuda::device::url_scheme )
            return region::cuda::device::bytes_from_url( u );
        else if ( scheme == region::managed::url_scheme )
            return region::managed::bytes_from_url( u );
        else if ( scheme == local::url_scheme )
            // Needs to stay in last position, because it will always return a url.
            return local::bytes_from_url( u );

        return make_unexpected(error::url_unknown_scheme);
    }

    result<url> url_from_bytes( span_cb bytes, bool allow_local ) {
        if (auto maybe_bytes = revert_convert_bytes(bytes); maybe_bytes)
            bytes = *maybe_bytes; // update the bytes from the reverted bytes and revert the convert_bytes function in bytes_from_url.

        //Could be handle by region::host but it's a bit cleaner to let managed handle itself.
        EMU_UNWRAP_RETURN_IF_TRUE(region::managed::url_from_bytes(bytes));

        EMU_UNWRAP_RETURN_IF_TRUE(region::host::url_from_bytes(bytes));

        EMU_UNWRAP_RETURN_IF_TRUE(region::cuda::device::url_from_bytes(bytes));

        // Should be dead last, because it will always return a url.
        if (allow_local)
            EMU_UNWRAP_RETURN_IF_TRUE(local::url_from_bytes(bytes));

        return make_unexpected( error::url_resource_not_registered );
    }

} // namespace sardine::detail
