#pragma once

#include <sardine/config.hpp>

#include <emu/type_traits.hpp>
#include <emu/fwd.hpp>
#include <emu/error.hpp>

#include <boost/url.hpp>
#include <boost/json.hpp>

namespace std
{

    struct error_code;

} // namespace std



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

    using boost::urls::url;
    using boost::urls::url_view;

namespace urls
{
    using boost::urls::params_view;
    using boost::urls::params_ref;

    using boost::urls::parse_uri;
    using boost::urls::parse_uri_reference;

} // namespace urls

} // namespace sardine
