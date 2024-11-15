#pragma once

#include <sardine/fwd.hpp>

#include <emu/concepts.hpp>
#include <emu/cstring_view.hpp>

namespace sardine::cpts
{

    template<typename T>
    concept url = std::same_as<T, sardine::url>
               or std::same_as<T, sardine::url_view>;

    template<typename T>
    concept url_params = std::same_as<T, sardine::urls::params_ref>
                      or std::same_as<T, sardine::urls::params_view>;

    template<typename T>
    concept url_of_aware = requires(const T& t) {
        { t.url() } -> cpts::url;
    };

    template<typename T>
    concept from_url_aware = requires(const sardine::url_view& url) {
        { T::from_url(url) };
    };

    // template<typename T>
    // concept loadable = requires(const boost::json::value& jv) {
    //     { T::load(jv) } -> std::same_as< result<T> >;
    // };

    template<typename T>
    concept view = emu::cpts::any_span<T>;

    template<typename T>
    concept mdview = emu::cpts::any_mdspan<T>;

    template<typename T>
    concept all_view = emu::is_ref<T> or std::is_pointer_v<T> or view<T> or mdview<T> or emu::cpts::any_string_view<T>;

    template<typename T>
    concept closable_lead_dim = requires(const T& t) {
        { t.close_lead_dim() };
    };

} // namespace sardine::cpts
