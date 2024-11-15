#pragma once

#include "buffer/impl.hpp"
#include <sardine/buffer/base.hpp>
#include <sardine/buffer/impl.hpp>
#include <sardine/buffer/ring.hpp>

namespace sardine
{

    template<typename Ctx, typename T>
    producer<std::decay_t<T>, Ctx> make_producer(T&& value) {
        return producer<std::decay_t<T>, Ctx>(EMU_FWD(value));
    }

    template<typename Ctx, typename T>
    consumer<std::decay_t<T>, Ctx> make_consumer(T&& value) {
        return consumer<std::decay_t<T>, Ctx>(EMU_FWD(value));
    }

    template <typename T, typename Ctx>
    box<T, Ctx>::box(interface_t view)
        : base_t(mapper_from(view))
        , prod_impl(buffer::make_s_producer<Ctx>(buffer::native::producer<host_context>{sardine::as_bytes(view)}))
        , cons_impl(buffer::make_s_consumer<Ctx>(buffer::native::consumer<host_context>{sardine::as_bytes(view)}))
        , storage(cons_impl->bytes())
        , value(storage.init(mapper().convert(cons_impl->bytes())))
    {}

    template <typename T>
    view_t<T>::view_t(interface_t view)
        : base_t(mapper_from(view))
        , impl(buffer::make_s_view(buffer::native::view_t{sardine::as_bytes(view)}))
    {}

    template <typename T, typename Ctx>
    producer<T, Ctx>::producer(interface_t view)
        : base_t(mapper_from(view))
        , impl(buffer::make_s_producer<Ctx>(buffer::native::producer<Ctx>{sardine::as_bytes(view)}))
    {}

    template <typename T, typename Ctx>
    consumer<T, Ctx>::consumer(interface_t view)
        : base_t(mapper_from(view))
        , impl(buffer::make_s_consumer<Ctx>(buffer::native::consumer<Ctx>{sardine::as_bytes(view)}))
    {}

    template<typename T, typename Ctx>
    auto factory<T, Ctx>::create_impl(url_view u) -> result<buffer::s_factory<Ctx>> {
        auto scheme = u.scheme();

        constexpr static auto requested_device_type = emu::location_type_of<T>::device_type;

        if (scheme == ring::url_scheme)
            return ring::factory<Ctx>::create(u, requested_device_type);
        else
            return buffer::bytes_factory<Ctx>::create(u, requested_device_type);
    }

    template<typename T, typename Ctx>
    auto factory<T, Ctx>::create(url_view u) -> result<factory> {
        // use scheme here for factory that need to know the actual type.
        // auto scheme = u.scheme();

        buffer::s_factory<Ctx> fac_impl = EMU_UNWRAP(create_impl(u));

        return mapper_from_mapping_descriptor<T>(fac_impl->mapping_descriptor())
            .map([&](auto mapper) {
                return factory{fac_impl, mapper};
            });
    }

    template<typename T, typename Ctx>
    auto box<T, Ctx>::open(url_view u) -> result<box> {
        return factory<T, Ctx>::create(u)
            .and_then([](auto f) -> result<box> {
                auto prod = EMU_UNWRAP(f.impl->create_producer());

                auto cons = EMU_UNWRAP(f.impl->create_consumer());

                return box<T, Ctx>(f.mapper, prod, cons);
            }
        );
    }

    template<typename T, typename Ctx>
    auto producer<T, Ctx>::open(url_view u) -> result<producer> {
        return factory<T, Ctx>::create(u)
            .and_then([](auto f) {
                return f.create_producer();
            }
        );
    }

    template<typename T, typename Ctx>
    auto consumer<T, Ctx>::open(url_view u) -> result<consumer> {
        return factory<T, Ctx>::create(u)
            .and_then([](auto f) {
                return f.create_consumer();
            }
        );
    }

    // template<typename T, typename Ctx>
    // auto factory<T, Ctx>::create(const json::value& jv) -> result<factory> {
    //     return json::factory<T, Ctx>::create(jv)
    //         .and_then([](auto&& p_impl) {
    //             return sardine::accessor<T>::create(*p_impl).map([&](auto accessor) {
    //                 return factory{p_impl, accessor};
    //             });
    //         });
    // }

    // // template<typename T>
    // // auto box<T>::load(const json::value& jv) -> result<box> {
    // //     return factory<T, host_context>::create(jv)
    // //         .and_then([](auto f) {
    // //             return f.create_box();
    // //         }
    // //     );
    // // }

    // template<typename T, typename Ctx>
    // auto consumer<T, Ctx>::load(const json::value& jv) -> result<consumer> {
    //     return factory<T, Ctx>::create(u)
    //         .and_then([](auto&& f) {
    //             return f.create_consumer();
    //         }
    //     );
    // }

    // template<typename T, typename Ctx>
    // auto producer<T, Ctx>::load(const json::value& jv) -> result<producer> {
    //     return factory<T, Ctx>::create(u)
    //         .and_then([](auto&& f) {
    //             return f.create_producer();
    //         }
    //     );
    // }

} // namespace sardine

template<typename T, typename Ctx, typename CharT>
struct fmt::formatter<sardine::box<T, Ctx>, CharT> {

    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const sardine::box<T, Ctx>& box, FormatContext& ctx) {
        return fmt::format_to(ctx.out(), "value({})", box.value);
    }
};

template<typename T, typename Ctx, typename CharT>
struct fmt::formatter<sardine::producer<T, Ctx>, CharT> {

    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const sardine::producer<T, Ctx>& value, FormatContext& ctx) {
        return fmt::format_to(ctx.out(), "producer({})", value.view());
    }
};

template<typename T, typename Ctx, typename CharT>
struct fmt::formatter<sardine::consumer<T, Ctx>, CharT> {

    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const sardine::consumer<T, Ctx>& value, FormatContext& ctx) {
        return fmt::format_to(ctx.out(), "consumer({})", value.view());
    }
};
