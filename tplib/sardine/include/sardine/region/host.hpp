#pragma once

#include <sardine/type.hpp>
#include <sardine/type/url.hpp>
#include <sardine/region/host/utility.hpp>

namespace sardine::region::host
{

    span_b open(string name);

    span_b create(string name, size_t size);

    span_b open_or_create(string name, size_t size);

    template<typename T>
    span<T> open(string name) {
        return emu::as_t<T>(open(std::move(name)));
    }

    template<typename T>
    span<T> create(string name, size_t size) {
        return emu::as_t<T>(create(std::move(name), size * sizeof(T)));
    }

    template<typename T>
    span<T> open_or_create(string name, size_t size) {
        return emu::as_t<T>(open_or_create(std::move(name), size * sizeof(T)));
    }

    optional<shm_handle> find_handle(const byte* ptr);

    constexpr auto url_scheme = "host";

    url url_from_unregistered_bytes(span_cb data);

    optional<result<url>> url_from_bytes(span_cb data);

    result<bytes_and_device> bytes_from_url(const url_view& u);

} // namespace sardine::region::host
