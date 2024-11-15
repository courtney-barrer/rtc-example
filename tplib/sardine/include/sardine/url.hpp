#pragma once

#include "type/url.hpp"
#include "utility.hpp"
#include <sardine/type.hpp>
#include <sardine/mapper.hpp>
#include <sardine/error.hpp>

#include <emu/location_policy.hpp>
#include <emu/dlpack.hpp>

namespace sardine
{

namespace detail
{

    result<url> url_from_bytes( span_cb bytes, bool allow_local = false );

    /// Returns bytes from url. Type info for special regions and bytes convertion.
    result<container_b> bytes_from_url( const url_view& u, emu::dlpack::device_type_t requested_dt);

    template<typename T>
    auto from_url( const url_view& u ) -> result<typename mapper<T>::view_type> {
        constexpr auto requested_dt = emu::location_type_of<T>::device_type;

        auto [view, capsule] = EMU_UNWRAP( detail::bytes_from_url( u, requested_dt ) ).as_pair();

        auto mapping = EMU_UNWRAP(make_mapping_descriptor(u.params()));

        // TODO: retrieve the capsule from bytes_from_url.
        return sardine::mapper_from_mapping_descriptor<T>(mapping, std::move(capsule))
            .map([&](auto mapper){
                return wrap_ref(mapper.convert(view));
            });
    }

    template<typename T>
    auto url_of(T&& value, bool allow_local) -> result<url> {

        auto bytes = sardine::as_bytes(value);

        auto url = EMU_UNWRAP(detail::url_from_bytes( bytes, allow_local ));

        //TODO: is it safe to maybe move value without invalidating bytes ?
        auto descriptor = sardine::mapper_from(value).mapping_descriptor();

        // managed_tensor_versioned will be deleted immediately after the call of update_url.
        update_url(url, descriptor);

        return url;
    }

} // namespace detail

    template<typename T>
    auto from_url( const url_view& u ) {
        if constexpr ( cpts::from_url_aware<T> )
            return T::from_url(u);
        else
            return detail::from_url<T>(u);
    }


    template<typename T>
    auto from_url_throw( const url_view& u ) {
        return EMU_UNWRAP_RES_OR_THROW(from_url<T>(u));
    }

    template<typename T>
    result<url> url_of(T&& value, bool allow_local = false) {
        if constexpr ( cpts::url_of_aware< emu::rm_cvref<T> > )
            return value.url();
        else
            return detail::url_of(EMU_FWD(value), allow_local);
    }

    template<typename T>
    result<url> url_of_throw(T&& value, bool allow_local = false) {
        return EMU_UNWRAP_RES_OR_THROW(url_of(EMU_FWD(value), allow_local));
    }

} // namespace sardine
