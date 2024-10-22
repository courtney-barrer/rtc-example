#pragma once

#include "type/url.hpp"
#include "utility.hpp"
#include <sardine/type.hpp>
#include <sardine/mapper.hpp>
#include <sardine/error.hpp>

#include <emu/location_policy.hpp>
#include <emu/dlpack.hpp>
#include <sardine/memory_converter.hpp>

namespace sardine
{

namespace detail
{

    result<url> url_from_bytes( span_cb bytes, bool allow_local = false );

    /// Returns bytes from url. No type info, no bytes convertion
    result<bytes_and_device> bytes_from_url( const url_view& u );

    inline auto byte_converter_for(emu::dlpack::device_type_t requested_dt) {
        return [requested_dt]( bytes_and_device res ) -> result< span_b > {
            if (requested_dt == res.device.device_type // The device type match
            or requested_dt == emu::dlpack::device_type_t::kDLExtDev) // the requested device type is unspecified
                return res.data;

            return convert_bytes(res, requested_dt);
        };
    }

    /// Returns bytes from url. Type info for special regions and bytes convertion.
    /// Types is needed for some sources such as embedded.
    template<typename T>
    result<span_b> bytes_from_url( const url_view& u ) {
        constexpr auto requested_dt = emu::location_type_of<T>::device_type;

        //here handle cases where opening url need to also know the type
        // but the result is still bytes.
        auto scheme = u.scheme();

        result<bytes_and_device> bytes_and_device;

        // if ( scheme == region::embedded::url_scheme )
        //     bytes_and_device = region::embedded::bytes_from_url<T>( u, requested_dt );
        // else
            bytes_and_device = detail::bytes_from_url(u );

        // Once we have the bytes, convert then if necessary.
        return bytes_and_device.and_then(byte_converter_for(requested_dt));
    }


    /// Returns bytes from url. Type info for special regions and bytes convertion.
    /// Types is needed for some sources such as embedded.
    // template<typename T>
    // result<emu::dlpack::scoped_tensor> tensor_from_url( const url_view& u ) {
    //     constexpr auto requested_dt = emu::location_type_of<T>::device_type;

    //     //here handle cases where opening url need to also know the type
    //     // but the result is still bytes.
    //     auto scheme = u.scheme();

    //     result<bytes_and_device> bytes_and_device;

    //     // if ( scheme == region::embedded::url_scheme )
    //     //     bytes_and_device = region::embedded::bytes_from_url<T>( u, requested_dt );
    //     // else
    //         bytes_and_device = detail::bytes_from_url(u );

    //     // Once we have the bytes, convert then if necessary.
    //     auto bytes = EMU_UNWRAP(bytes_and_device.and_then(byte_converter_for(requested_dt)));

    //     return dlpack::from_url(bytes, u);
    // }

} // namespace detail

    template<typename T>
        // requires (cpts::mappable<T> or std::same_as<T, url>)
    auto from_url( const url_view& u ) {
        if constexpr (std::same_as<T, url>)
            // Special case when explicitly asks for a url
            // we return what we have regardless of if there is a url or not.
            //Note: maybe we could open the url as a string_view ?
            return result<url>{ url{u} };
        else {
            // auto tensor = EMU_UNWRAP(tensor_from_url<T>(u));

            // if constexpr (emu::dlpack::keep_capsule<T>)
            //     EMU_INVOKE_AT_SCOPE_EXIT([&]{
            //         // Put the data capsule in a sink alive forever.
            //         detail::keep_alive(std::move(tensor).data_capsule());
            //     });
            return [&]() -> result<typename mapper<T>::view_type>
            {
                auto view = EMU_UNWRAP( detail::bytes_from_url<T>( u ) );
                // return emu::dlpack::import_from_scoped_tensor<T>(tensor);

                auto mapping = EMU_UNWRAP(make_mapping_descriptor(u.params()));

                return sardine::mapper_from_mapping_descriptor<T>(mapping, emu::capsule())
                    .map([&](auto mapper){
                        return wrap_ref(mapper.convert(view));
                    });
            }();


            // return make_mapper< T >(u.params())
            //     .map([&view](auto as) {
            //         return wrap_ref(as.convert(view));
            //     });
            // }(); // immediately call.
        }
    }

    template<typename T>
    result<url> url_of(T&& value, bool allow_local = false) {
        if constexpr ( cpts::url_aware<T> )
            return value.url();
        else {
            auto bytes = sardine::as_bytes(value);

            auto url = EMU_UNWRAP(detail::url_from_bytes( bytes, allow_local ));

            //TODO: is it safe to maybe move value without invalidating bytes ?
            auto descriptor = sardine::mapper_from(value).mapping_descriptor();

            // managed_tensor_versioned will be deleted immediately after the call of update_url.
            update_url(url, descriptor);

            return url;
        }
    }

} // namespace sardine
