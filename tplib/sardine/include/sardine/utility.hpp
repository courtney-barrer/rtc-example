#pragma once

#include <sardine/type.hpp>

#include <emu/span.hpp>
#include <emu/capsule.hpp>
#include <emu/cstring_view.hpp>

#include <fmt/format.h>

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/hana/core/make.hpp>

#include <type_traits>
#include <stdexcept>
#include <ranges>

// namespace nanobind {
//     template <typename... Args> class ndarray;
// }

namespace sardine
{
    template<typename T>
    inline byte* b_ptr_of(const T* t) {
        return const_cast<byte*>(reinterpret_cast<const byte*>(t));
    }

    template<typename T>
    inline byte* b_ptr_of(std::span<T> s) {
        return b_ptr_of(s.data());
    }

    template<typename T>
    inline void* v_ptr_of(const T* t) {
        return const_cast<void*>(reinterpret_cast<const void*>(t));
    }

    template<typename T>
    inline void* v_ptr_of(std::span<T> s) {
        return v_ptr_of(s.data());
    }

    template<typename T>
    constexpr auto wrap_ref(T&& t) noexcept {
        if constexpr (emu::is_ref<T>)
            return std::ref(t);
        else
            return t;
    }

    // template<typename T>
    // constexpr result<T> as_result(myboost::system::result<T> res) {
    //     if (res)
    //         return wrap_ref(res.value());
    //     else
    //         return unexpected(res.error());
    // }

    inline void remove(cstring_view name) {
        myboost::interprocess::shared_memory_object::remove(name.c_str());
    }

    // auto split_string(std::string_view words, std::string_view delim = " ") {
    //     return std::views::split(words, delim) | std::views::transform([](auto sr) {
    //         return std::string_view(&*sr.begin(), std::ranges::distance(sr));
    //     });
    // }

    // Handle reference, span and mdspan
    // converts a value to a span of bytes, automatically detect constness

namespace detail
{
        template<typename View>
        struct as_span_of_bytes
        {

            template<typename ActualView>
            static constexpr auto as_span(ActualView&& value) {

                if constexpr (emu::cpts::span< View >) {
                    return emu::as_auto_bytes(value);
                } else if constexpr (emu::cpts::mdspan< View >) {
                    return emu::as_auto_bytes(std::span{value.data_handle(), value.mapping().required_span_size()});
                } else {
                    return emu::as_auto_bytes(std::span{&value, 1});
                }
            }
        };

        template<emu::cpts::any_string_view Str>
        struct as_span_of_bytes<Str> {
            static constexpr auto as_span(const Str& value) -> span_cb {
                return std::as_bytes(std::span(value));
            }
        };

        // template<typename View>
            // requires (not std::same_as<emu::decay<View>, nanobind::ndarray<>>)

} // namespace detail



    template<typename T>
    constexpr auto as_span_of_bytes(T&& value) {
        return detail::as_span_of_bytes< emu::rm_cvref<T> >::as_span(std::forward<T>(value));
    }


namespace detail
{

    template<typename T, typename Tuple>
    struct is_constructible_from_tuple_args : std::false_type {};

    template<typename T, typename ...Args>
    struct is_constructible_from_tuple_args<T, std::tuple<Args...>> : std::is_constructible<T, Args...> {};

} // namespace detail

    template<typename T, typename ...Args>
    constexpr bool is_constructible_from_tuple_args_v = detail::is_constructible_from_tuple_args<T, Args...>::value;

    template<typename Map, typename Key, typename Fn>
    auto auto_emplace(Map & map, const Key & key, Fn && fn) {

        using result_type = std::invoke_result_t<Fn>;

        if constexpr (std::is_constructible_v<typename Map::mapped_type, result_type>)
            return map.emplace(key, fn());
        else
        {
            static_assert(
                is_constructible_from_tuple_args_v<typename Map::mapped_type, result_type>,
                "Mapped type is not constructible from the result of the function"
            );

            return map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(key),
                fn());
        }
    }

    template<typename Map, typename Key, typename Fn>
    auto find_or_emplace(Map & map, const Key & key, Fn && fn)
    {
        auto it = map.find(key);
        if (it == map.end())
            it = auto_emplace(map, key, fn).first;
        return it;
    }

    template<typename Map, typename Key, typename Fn>
    auto emplace_or_throw(Map & map, const Key & key, Fn && fn)
    {
        auto it = map.find(key);
        if (it != map.end())
            throw std::runtime_error("Key already exists");
        return auto_emplace(map, key, fn).first;
    }

namespace detail
{
    void keep_alive(emu::capsule c);

} // namespace detail

    using myboost::hana::make;

} // namespace sardine
