#pragma once

#include <sardine/type.hpp>
#include <sardine/type/url.hpp>
#include <sardine/type/json.hpp>
#include <sardine/mapper.hpp>

#include <emu/capsule.hpp>

#include <boost/json/value_to.hpp>

namespace sardine::region::embedded
{

    constexpr auto data_key       = "json";

    // optional<shm_handle> find_handle(const byte* ptr);

    constexpr auto url_scheme = "embedded";

    // optional<result<url>> url_from_bytes(span_cb data);

namespace detail
{
    void keep_alive(emu::capsule c);

} // namespace detail

namespace spe
{

    template<typename T>
    struct default_json_adaptor
    {
        using json_to_type = T;
        using json_from_type = T;

        static result<span_b> value_to(const json::value& v, emu::dlpack::device_type_t /* requested_dt */) {
            //Note: for now, we ignore the requested_dt. Maybe one day, try to instantiate
            //the data on the correct device_type.
            auto data = EMU_UNWRAP_OR_RETURN_ERROR(json::try_value_to< json_to_type >(v));

            // move data on heap, the only way to guarantee the move won't affect the bytes bellow.
            auto parsed_data = std::make_unique<json_to_type>(std::move(data));

            auto bytes = as_span_of_bytes(*parsed_data);

            // move the unique pointer in the cache. Does not affect the position of the data on the heap.
            detail::keep_alive(emu::capsule(std::move(parsed_data)));

            return bytes;
        }

        static json::value value_from(const T& value) {
            return json::value_from(value);
        }
    };


    template<emu::cpts::any_span Span> // including cuda::device::span. Need to assess that.
    struct default_json_adaptor<Span>
    {
        using span_t = Span;
        using value_type = typename span_t::value_type;

        using json_to_type = std::vector< value_type >;
        using json_from_type = span_t;

        static result<span_b> value_to(const json::value& v, emu::dlpack::device_type_t /* requested_dt */) {
            //Note: for now, we ignore the requested_dt. Maybe one day, try to instantiate
            //the data on the correct device_type.
            auto data = EMU_UNWRAP_OR_RETURN_ERROR(json::try_value_to< json_to_type >(v));

            auto bytes = as_span_of_bytes(std::span{data});

            // move the unique pointer in the cache. Does not affect the position of the data on the heap.
            detail::keep_alive(emu::capsule(std::move(data)));

            return bytes;
        }

        static json::value value_from(const span_t& span) {
            return json::value_from(span);
        }
    };



    template<emu::cpts::any_string_view StringView> // including cuda::device::span. Need to assess that.
    struct default_json_adaptor<StringView> : default_json_adaptor< std::span<typename StringView::value_type> >
    {
        using string_view_t = StringView;
        using value_type = typename string_view_t::value_type;
        using traits_type = typename string_view_t::traits_type;

        using json_to_type = std::basic_string<value_type, traits_type>;
        using json_from_type = string_view_t;

        static result<span_b> value_to(const json::value& v, emu::dlpack::device_type_t /* requested_dt */) {
            //Note: for now, we ignore the requested_dt. Maybe one day, try to instantiate
            //the data on the correct device_type.
            auto data = EMU_UNWRAP_OR_RETURN_ERROR(json::try_value_to< json_to_type >(v));

            // move data on heap, the only way to guarantee thestring SBO and move won't affect the bytes bellow.
            auto parsed_data = std::make_unique<json_to_type>(std::move(data));

            auto bytes = as_span_of_bytes(std::span{*parsed_data});

            // move the unique pointer in the cache. Does not affect the position of the data on the heap.
            detail::keep_alive(emu::capsule(std::move(parsed_data)));

            return bytes;
        }

        static json::value value_from(const string_view_t& span) {
            return json::value_from(span);
        }
    };


    template<typename T>
    struct json_adaptor : default_json_adaptor<T>
    {};

} // namespace spe


    template<typename T>
        requires ( not cpts::has_json_to<T> )
    result<bytes_and_device> bytes_from_url(url_view u, emu::dlpack::device_type_t requested_dt) {
        return make_unexpected(error::embedded_type_not_handled);
    }

    template<cpts::has_json_to T>
    result<bytes_and_device> bytes_from_url(url_view u, emu::dlpack::device_type_t requested_dt) {
        // parse the url as a json to create the type T and keep it alive.
        auto params = u.params();

        auto json_data = EMU_UNWRAP_OR_RETURN_UNEXPECTED(urls::try_parse_at<json::value>(params, data_key),
                                                         error::embedded_url_missing_json);

        auto bytes = EMU_UNWRAP(spe::json_adaptor<T>::value_to(json_data, requested_dt));

        return bytes_and_device{
            .region=bytes,
            .data=bytes,
            .device={
                .device_type=emu::dlpack::device_type_t::kDLCPU,
                .device_id=0
            }
        };

    }

    template<cpts::has_json_from T>
    url url_of(const T& value) {
        auto json_value = spe::json_adaptor<T>::value_from(value);

        auto result = url()
            .set_scheme(url_scheme)
            .set_params({{data_key, boost::json::serialize(json_value)}});

        sardine::update_url( result, value);

        return result;
    }

} // namespace sardine::region::embedded
