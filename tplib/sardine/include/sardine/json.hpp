#pragma once

#include <sardine/type.hpp>
#include <sardine/mapper.hpp>
#include <sardine/error.hpp>
#include <sardine/memory_converter.hpp>

#include <emu/location_policy.hpp>
#include <emu/pointer.hpp>

namespace sardine
{

    template<typename T>
    auto from_json( const json::value& value ) -> result<T> {
        // First, we check if it is an object that olds "_url" field.
        // Then because it is too easy to accidentally cast something into a url
        // we try to cast to the object.
        // If it fails, then we try to cast the object into a url and open it.


        // first try to check if its an object that contains "_url" field.
        // then check if the value is itself parsable as url.
        if (auto maybe_url = json::opt_to<url>(value, "_url"); maybe_url) {
            // Have to unwrap because from_url and from_json return type are slightly different.
            return EMU_UNWRAP(from_url<T>(*maybe_url));
        }

        // Then try to cast the json to the object if possible
        if constexpr (cpts::has_json_to<T>)
            EMU_UNWRAP_RETURN_IF_TRUE(json::try_value_to<T>(value));

        // Finally, try to parse json to url and open it.
        if ( auto maybe_url = json::opt_to<url>(value); maybe_url )
            return from_url<T>(*maybe_url);

        return make_unexpected(error::json_invalid_json);

    }

} // namespace sardine
