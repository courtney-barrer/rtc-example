#pragma once

#include <sardine/type/json.hpp>
#include <sardine/url.hpp>

namespace sardine
{

    template <typename T>
    result<T> load( const json::value& jv ) {
        if constexpr (cpts::loadable<T>)
            return T::load(jv);

        // Create the variable with an error state (even if it is not an error)
        result<url> maybe_url = make_unexpected(error::success);

        // Try to convert the json into a url or check if it contains a one.

        // Special case when requesting a std::string, we don't want to accidentally parse it to a url.
        // url support is done by explicitly putting it inside jv["_url"]
        if constexpr (not std::is_same_v<T, std::string>)
            maybe_url = json::try_value_to< url >( jv );

        // If there is not url, we check if the json has a url inside "_url" key.
        if (not maybe_url and jv.is_object()) {
            auto& obj = jv.get_object();
            if (auto* u = obj.if_contains("_url"); u)
                maybe_url = json::try_value_to< url >( *u );
        }

        EMU_RETURN_IF_VALUE(maybe_url.map( &from_url< T > ));

        if constexpr ( not emu::is_ref<T> and json::bj::has_value_to<T>::value )
            return json::try_value_to<T>( jv );
        else
            return make_unexpected( error::json_parse_reference );

    }

} // namespace sardine
