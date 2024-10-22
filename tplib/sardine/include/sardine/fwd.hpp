#pragma once

#include <emu/type_traits.hpp>
#include <emu/error.hpp>

namespace std
{

    struct error_code;

} // namespace std


namespace myboost::urls
{

    struct url;
    struct url_view;

    struct params_view;
    struct params_ref;

    struct parse_uri;
    struct parse_uri_reference;

} // namespace myboost::urls

namespace myboost::json
{

    struct value;


} // namespace myboost::json


namespace tl
{

    template<typename T, typename E>
    struct expected;

} // namespace tl


namespace sardine
{
    // template<typename T>
    // using return_type = std::conditional_t<emu::is_ref<T>, std::reference_wrapper<std::remove_reference_t<T>>, T>;

    using emu::result;

    using myboost::urls::url;
    using myboost::urls::url_view;

namespace urls
{
    using myboost::urls::params_view;
    using myboost::urls::params_ref;

    using myboost::urls::parse_uri;
    using myboost::urls::parse_uri_reference;

} // namespace urls

} // namespace sardine
