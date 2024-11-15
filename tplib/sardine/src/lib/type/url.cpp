#include <sardine/type/url.hpp>

#include <sardine/type/json.hpp>

namespace EMU_BOOST_NAMESPACE::urls
{

    void tag_invoke( json::value_from_tag, json::value& jv, const url_view& url ) {
        jv = url.buffer();
    }

    json::result< url > tag_invoke( json::try_value_to_tag< url >, const json::value& jv ) {
        auto* str = jv.if_string();

        if (! str) return json::make_error_code( json::error::not_string );

        return sardine::urls::parse_uri_reference(*str);
    }

    void tag_invoke( json::value_from_tag, json::value& jv, const params_view& params ) {
        auto& obj = jv.as_object();
        for (auto&& [key, value, has_value] : params)
            if (not has_value)
                obj[key] = nullptr;
            else
                obj[key] = value;

    }

} // namespace EMU_BOOST_NAMESPACE::urls
