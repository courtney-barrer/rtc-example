#pragma once

#include <sardine/type.hpp>
#include <sardine/type/url.hpp>

namespace sardine::cuda::device
{

    constexpr auto url_scheme = "cudadevice";

    optional<result<url>> url_from_bytes(span_cb data);

    result<bytes_and_device> bytes_from_url(url_view u);

} // namespace sardine::cuda::device
