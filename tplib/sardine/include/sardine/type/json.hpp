#pragma once

#include <sardine/utility.hpp>

#include <emu/macro.hpp>
#include <emu/utility.hpp>

#include <boost/json.hpp>
#include <boost/callable_traits/args.hpp>
#include <boost/callable_traits/return_type.hpp>

namespace sardine
{

namespace cpts
{

    template<typename T>
    concept has_json_to = boost::json::has_value_to< T >::value;

    template<typename T>
    concept has_json_from = boost::json::has_value_from< T >::value;

} // namespace cpts

namespace json
{

    //TODO: use specific types instead
    namespace bj = boost::json;
    using bj::value;
    using bj::object;
    using bj::array;
    using bj::string;

    using bj::serialize;
    using bj::parse;

    using bj::value_to;
    using bj::value_from;
    using bj::try_value_to;

    using bj::value_to_tag;

    // template<typename Value>
    // optional<object> try_object(Value&& value) {
    //     if (value.is_object())
    //         return EMU_FWD(value).as_object();
    //     else
    //         return nullopt;
    // }

    // template<typename Value>
    // optional<array> try_array(Value&& value) {
    //     if (value.is_array())
    //         return EMU_FWD(value).as_array();
    //     else
    //         return nullopt;
    // }

    // template<typename Value>
    // optional<string> try_string(Value&& value) {
    //     if (value.is_string())
    //         return EMU_FWD(value).as_string();
    //     else
    //         return nullopt;
    // }

    // template<typename Fn, std::convertible_to<string_view>... Names>
    // constexpr auto try_find_all(Fn&& f, const object& obj, Names&&... names) {
    //     namespace ct = boost::callable_traits;

    //     return [&] (auto... its) -> result< ct::return_type_t<Fn> > {
    //         if (((its != obj.end()) && ...))
    //             return f(its->value()...);
    //         else
    //             return bj::make_error_code( bj::error::not_found );
    //     } (obj.find(names)...);
    // }

    // template<typename Fn, std::convertible_to<string_view>... Names>
    // auto try_invoke_if_all(Fn&& f, const object& obj, Names&&... names) {
    //     namespace ct = boost::callable_traits;

    //     return try_find_all([&f] (const auto&... values) -> result< ct::return_type_t<Fn> > {
    //         return emu::invoke_with_args([&]<typename... Ts> (emu::type_pack<Ts...>) {
    //             return [&f] (auto&&... values) -> result< ct::return_type_t<Fn> > {
    //                 if((values.has_value() && ...))
    //                     return f(EMU_FWD(values).value()...);
    //                 else
    //                     return emu::get_first_error(as_result(EMU_FWD(values))...);
    //             } (bj::try_value_to<Ts>(values)...);
    //         }, /*reference= */ f);
    //     }, obj, EMU_FWD(names)...);
    // }

    // template<typename T>
    // result<T> try_value_to(const value& j) {
    //     return as_result(bj::try_value_to<T>(j));
    // }


    template<typename T>
    T object_to(const object& obj, string_view key) {
        return value_to<T>(obj.at(key));
        // const value* it = obj.if_contains(key);

        // EMU_TRUE_OR_RETURN_NULLOPT(it);

        // return opt_to<T>(*it);
    }

    template<typename T>
    optional<T> opt_to(const value& j) {
        auto res = bj::try_value_to<T>(j);

        EMU_TRUE_OR_RETURN_NULLOPT(res.has_value());

        return res.value();
    }

    template<typename T>
    optional<T> opt_to(const object& obj, string_view key) {
        const value* it = obj.if_contains(key);

        EMU_TRUE_OR_RETURN_NULLOPT(it);

        return opt_to<T>(*it);
    }

    template<typename T>
    optional<T> opt_to(const array& arr, size_t index) {
        const value* it = arr.if_contains(index);

        EMU_TRUE_OR_RETURN_NULLOPT(it);

        return opt_to<T>(*it);
    }

    template<typename T>
    optional<T> opt_to(const value& val, string_view key) {
        // try to cast val into object. If so, try access value at key.
        return opt_to<T>(EMU_UNWRAP_OR_RETURN_NULLOPT(val.if_object()), key);
    }

    template<typename T>
    optional<T> opt_to(const value& val, size_t index) {
        // try to cast val into object. If so, try access value at key.
        return opt_to<T>(EMU_UNWRAP_OR_RETURN_NULLOPT(val.if_array()), index);
    }

    template<typename T>
    T value_or(const value& j, T&& def) {
        return opt_to< std::decay_t<T> >(j).value_or(EMU_FWD(def));
    }

    template<typename T>
    T value_or(const object& j, string_view key, T&& def) {
        return opt_to< std::decay_t<T> >(j, key).value_or(EMU_FWD(def));
    }

    /**
     * @brief Try to convert a value to an object, access a key and convert it to T.
     * If any of the steps fail, return def.
     *
     * @tparam T
     * @param j
     * @param key
     * @param def
     * @return T
     */
    template<typename T>
    T value_or(const value& j, string_view key, T&& def) {
        return emu::as_opt(j.if_object()).and_then([&] (const object& obj) {
            return opt_to< std::decay_t<T> >(obj, key);
        }).value_or(EMU_FWD(def));
    }

} // namespace json

} // namespace sardine

// namespace emu
// {

//     template<typename T>
//     result_for< ip_address, value >::type
//     tag_invoke( const try_value_to_tag< ip_address >&, value const& jv )
//     {
//         if( !jv.is_array() )
//             return make_error_code( std::errc::invalid_argument );

//         array const& arr = jv.get_array();
//         if( arr.size() != 4 )
//             return make_error_code( std::errc::invalid_argument );

//         boost::system::result< unsigned char > oct1
//             = try_value_to< unsigned char >( arr[0] );
//         if( !oct1 )
//             return make_error_code( std::errc::invalid_argument );

//         boost::system::result< unsigned char > oct2
//             = try_value_to< unsigned char >( arr[1] );
//         if( !oct2 )
//             return make_error_code( std::errc::invalid_argument );

//         boost::system::result< unsigned char > oct3
//             = try_value_to< unsigned char >( arr[2] );
//         if( !oct3 )
//             return make_error_code( std::errc::invalid_argument );

//         boost::system::result< unsigned char > oct4
//             = try_value_to< unsigned char >( arr[3] );
//         if( !oct4 )
//             return make_error_code( std::errc::invalid_argument );

//         return ip_address{ *oct1, *oct2, *oct3, *oct4 };
//     }


// } // namespace emu
