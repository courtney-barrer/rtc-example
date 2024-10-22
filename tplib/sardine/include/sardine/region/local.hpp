#pragma once

#include <sardine/type.hpp>
#include <sardine/type/url.hpp>

namespace sardine::local
{


    constexpr auto url_scheme = "local";

    // template<typename T>
    // optional<url> url_of(const T& value) {
    //     return detail::url_from_bytes(as_span(value)).map([&](auto url) {
    //         sardine::update_url(url, value);
    //         return url;
    //     });
    // }
    optional<result<url>> url_from_bytes(span_cb data);

    result<bytes_and_device> bytes_from_url(url_view u);

} // namespace sardine::local
