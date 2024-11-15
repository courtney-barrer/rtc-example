#pragma once

#include <sardine/config.hpp>
#include <sardine/concepts.hpp>
#include <sardine/type.hpp>
#include <sardine/type/json.hpp>

// #include <emu/detail/dlpack_types.hpp>
// #include <emu/dlpack.hpp>
#include <emu/charconv.hpp>

#include <boost/url.hpp>
#include <boost/lexical_cast.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

namespace sardine
{

    using boost::urls::url;
    using boost::urls::url_view;

namespace urls
{
    using boost::urls::param_view;

    using boost::urls::params_view;
    using boost::urls::params_ref;

    using boost::urls::parse_uri;
    using boost::urls::parse_uri_reference;

    using boost::urls::ignore_case;


    inline optional<std::string> try_get_at(params_view params, std::string_view key) {
        if (auto it = params.find(key); it != params.end()) {
            const auto& param = *it;
            if ( not param.has_value )
                return std::string{};
            else
                return std::string{param.value};
        }
        return nullopt;
    }


    template<typename T>
    optional<T> try_parse_at(params_view params, std::string_view key) {
        if (auto it = params.find(key); it != params.end()) {
            const auto& param = *it;
            if ( not param.has_value )
                return T{};
            else
                return json::opt_to<T>(json::parse(param.value));
        }
        return nullopt;
    }

    template<typename T>
    inline result<T> try_parse_number_at(params_view params, std::string_view key, T default_) {
        if (auto str = try_get_at(params, key); str) {
            T result;
            EMU_TRUE_OR_RETURN_ERROR(emu::from_chars(*str, result));
            return result;
        } else {
            return default_;
        }
    }

    template<typename T>
    inline result<T> try_parse_number_at(params_view params, std::string_view key) {
        if (auto str = try_get_at(params, key); str) {
            T result;
            EMU_TRUE_OR_RETURN_ERROR(emu::from_chars(*str, result));
            return result;
        } else {
            return make_unexpected(error::url_param_not_found);
        }
    }

} // namespace urls


// namespace dlpack
// {


//     constexpr auto shape_key = "shape";
//     constexpr auto stride_key = "strides";
//     constexpr auto byte_offset_key = "boffset";
//     constexpr auto read_only_key = "boffset";

    // void update_url(url& u, const emu::dlpack::managed_tensor_versioned_t& mtv) {
    //     auto params = u.params();

    //     const auto& tensor = mtv.dl_tensor;

    //     const auto& device = tensor.device;
    //     // if device_type is CPU, do not bother put that in the URL.
    //     if (device.device_type != kDLCPU) {
    //         auto device_type = static_cast<std::underlying_type_t<emu::dlpack::device_type_t>>(device.device_type);
    //         params.set(device_type_key, fmt::to_string(device_type));
    //         params.set(device_id_key, fmt::to_string(device.device_id));
    //     }

    //     const auto& dtype = tensor.dtype;
    //     params.set(data_type_code_key, fmt::to_string(dtype.code));
    //     params.set(bits_key, fmt::to_string(dtype.bits));
    //     // if dtype is not vectorized, do not bother specifying lanes number.
    //     if (dtype.lanes != 1)
    //         params.set(lanes_key, fmt::to_string(dtype.lanes));

    //     if (tensor.ndim > 0) {
    //         params.set(shape_key, fmt::format("[{}]", fmt::join(tensor.shape, tensor.shape + tensor.ndim, ",")));

    //         if (tensor.strides != NULL)
    //             params.set(stride_key, fmt::format("[{}]", fmt::join(tensor.strides, tensor.strides + tensor.ndim, ",")));
    //     }

    //     if(emu::dlpack::flag_read_only(mtv.flags))
    //         // set the key with no value.
    //         params.append({read_only_key, nullptr});
    // }

//     result<emu::dlpack::scoped_tensor> from_url(span_b data, const url_view& u) {
//         auto params = u.params();

//         using namespace emu::dlpack;

//         device_type_t device_type = static_cast<device_type_t>(EMU_UNWRAP(urls::try_parse_number_at(params, device_type_key, static_cast<device_type_under_t>(kDLCPU))));
//         int32_t device_id = EMU_UNWRAP(urls::try_parse_number_at(params, device_id_key, int32_t(0)));

//         device_t device{device_type, device_id};

//         uint8_t code = EMU_UNWRAP(urls::try_parse_number_at<uint8_t>(params, data_type_code_key));
//         uint8_t bits = EMU_UNWRAP(urls::try_parse_number_at<uint8_t>(params, bits_key));
//         uint16_t lanes = EMU_UNWRAP(urls::try_parse_number_at(params, bits_key, uint16_t(1)));

//         data_type_t data_type{code ,bits, lanes};

//         using vector_t = std::vector<int64_t>;

//         emu::container<int64_t> shape = urls::try_parse_at<vector_t>(params, shape_key).value_or(vector_t{});
//         emu::container<int64_t> strides = urls::try_parse_at<vector_t>(params, stride_key).value_or(vector_t{});

//         uint64_t flags = (params.contains(read_only_key) ? DLPACK_FLAG_BITMASK_READ_ONLY : 0);

//         return emu::dlpack::scoped_tensor{data.data(), device, data_type, 0, {}, shape, strides, flags};
//     }

// } // namespace dlpack



} // namespace sardine

namespace EMU_BOOST_NAMESPACE::urls
{

    void tag_invoke( json::value_from_tag, json::value& jv, const url_view& url );

    json::result< url > tag_invoke( json::try_value_to_tag< url >, const json::value& jv );

    void tag_invoke( json::value_from_tag, json::value& jv, const params_view& params );

    // We are only interested by converting to json.
    json::result< params_ref > tag_invoke( json::try_value_to_tag< params_ref >, const json::value& jv ) = delete;

} // namespace EMU_BOOST_NAMESPACE::urls

template <sardine::cpts::url Url> struct fmt::formatter<Url> : ostream_formatter {};

template<typename CharT>
struct fmt::formatter<EMU_BOOST_NAMESPACE::core::basic_string_view<CharT>> : fmt::formatter<fmt::basic_string_view<CharT>> {
    template <typename FormatContext>
    auto format(EMU_BOOST_NAMESPACE::core::basic_string_view<CharT> data, FormatContext &ctx) const {
        return fmt::formatter<fmt::basic_string_view<CharT>>::format({data.data(), data.size()}, ctx);
    }
};
