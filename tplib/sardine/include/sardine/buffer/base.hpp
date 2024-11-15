#pragma once

#include <sardine/type.hpp>
#include <sardine/url.hpp>
#include <sardine/context.hpp>
#include <sardine/mapper.hpp>
#include <sardine/buffer/utility.hpp>
#include <sardine/buffer/interface.hpp>
#include <sardine/mapper/mapper_base.hpp>

#include <emu/concepts.hpp>

#include <cstddef>

namespace sardine
{

    template<typename T, typename Ctx = host_context>
    struct box : buffer::mapper_base< T, box< T, Ctx > >
    {
        using base_t = buffer::mapper_base< T, box >;
        using base_t::mapper_t;
        using base_t::mapper;

        using interface_t = base_t::view_type;

        buffer::s_producer<Ctx> prod_impl;
        buffer::s_consumer<Ctx> cons_impl;

        // storage may be empty if the T is not a view.
        [[no_unique_address]] buffer::storage<T> storage;
        T value;


        box(interface_t view);

        box( sardine::mapper<T> a, buffer::s_producer<Ctx> p_impl, buffer::s_consumer<Ctx> c_impl )
            : base_t(std::move(a))
            , prod_impl(std::move(p_impl))
            , cons_impl(std::move(c_impl))
            , storage(cons_impl->bytes())
            , value(storage.init(this->convert(cons_impl->bytes())))
        {}

        void send(Ctx& ctx) {
            // TODO: move the copy out of the accessor and add the Ctx as a parameter.
            buffer::copy(value, this->convert(prod_impl->bytes()), ctx);
            prod_impl->send(ctx);
        }
        void recv(Ctx& ctx) {
            cons_impl->recv(ctx);
            buffer::copy(this->convert(cons_impl->bytes()), value, ctx);
        }

        void revert(Ctx& ctx) {
            prod_impl->revert(ctx);
            cons_impl->revert(ctx);
        }

        sardine::url url() const {
            auto u = cons_impl->url();
            update_url(u, mapper());
            return u;
        }

        template<typename NT>
        auto clone_with_new_mapper(sardine::mapper<NT> new_accessor) const -> box<NT, Ctx> {
            return box<NT, Ctx>( new_accessor, prod_impl, cons_impl );
        }

        // static auto load(url_view u/* , map_t parameter = map_t() */) -> result<box>;
        static auto open(url_view u) -> result<box>;
    };

    template<typename T>
    struct view_t : buffer::mapper_base < T, view_t <T> >
    {
        using base_t = buffer::mapper_base< T, view_t >;
        using base_t::mapper_t;
        using base_t::mapper;

        using interface_t = base_t::view_type;

        buffer::s_view impl;

        view_t(interface_t view);

        view_t(sardine::mapper<T> a, buffer::s_view i)
            : base_t(std::move(a))
            , impl(std::move(i))
        {}

        auto view() const -> decltype(auto) {
            return this->convert(impl->bytes());
        }

        sardine::url url() const {
            auto url = impl->url();
            update_url(url, mapper());
            return url;
        }

        template<typename NT>
        auto clone_with_new_mapper(sardine::mapper<NT> new_accessor) const -> view_t<NT> {
            return view_t<NT>( new_accessor, impl );
        }

    };

    template <typename T, typename Ctx = host_context>
    struct producer : buffer::mapper_base< T, producer< T, Ctx > >
    {
        using base_t = buffer::mapper_base< T, producer >;
        using base_t::mapper_t;
        using base_t::mapper;

        using interface_t = base_t::view_type;

        buffer::s_producer<Ctx> impl;

        producer(interface_t view);

        producer(sardine::mapper<T> a, buffer::s_producer<Ctx> i)
            : base_t(std::move(a))
            , impl(std::move(i))
        {}

        auto view() const -> decltype(auto) {
            return this->convert(impl->bytes());
        }

        void send(Ctx& ctx) { impl->send(ctx); }
        void revert(Ctx& ctx) { impl->revert(ctx); }

        // static auto load(const json::value& jv) -> result<producer>;

        sardine::url url() const {
            auto url = impl->url();
            update_url(url, mapper());
            return url;
        }

        template<typename NT>
        auto clone_with_new_mapper(sardine::mapper<NT> new_accessor) const -> producer<NT, Ctx> {
            return producer<NT, Ctx>( new_accessor, impl );
        }

        // sardine::url url() const { return impl->url(); }

        // static auto open(json_t json, map_t parameter = map_t()) -> result<producer>;
        static auto open(url_view u) -> result<producer>;
    };

    template <typename T, typename Ctx = host_context>
    struct consumer : buffer::mapper_base< T, consumer< T, Ctx > >
    {
        using base_t = buffer::mapper_base< T, consumer >;
        using base_t::mapper_t;
        using base_t::mapper;

        using interface_t = base_t::view_type;

        buffer::s_consumer<Ctx> impl;

        consumer(interface_t view);

        consumer(sardine::mapper<T> a, buffer::s_consumer<Ctx> i)
            : base_t(std::move(a))
            , impl(std::move(i))
        {}

        auto view() const -> decltype(auto) {
            return this->convert(impl->bytes());
        }

        void recv(Ctx& ctx) { impl->recv(ctx); }
        void revert(Ctx& ctx) { impl->revert(ctx); }

        static auto load(const json::value& jv) -> result<consumer>;

        sardine::url url() const {
            auto url = impl->url();
            update_url(url, mapper());
            return url;
        }


        template<typename NT>
        auto clone_with_new_mapper(sardine::mapper<NT> new_accessor) const -> consumer<NT, Ctx> {
            return consumer<NT, Ctx>( new_accessor, impl );
        }
        // sardine::url url() const { return impl->url(); }

        // static auto open(json_t json, map_t parameter = map_t()) -> result<consumer>;
        static auto open(url_view u) -> result<consumer>;

    };

    template<typename T, typename Ctx = host_context>
    struct factory
    {

        buffer::s_factory<Ctx> impl;
        [[no_unique_address]] sardine::mapper<T> mapper;
        default_mapping_descriptor mapping_desc;


        factory(buffer::s_factory<Ctx> impl, sardine::mapper<T> mapper)
            : impl(std::move(impl))
            , mapper(std::move(mapper))
            , mapping_desc(this->mapper.mapping_descriptor())
        {}

        static auto create_impl(url_view u) -> result< buffer::s_factory<Ctx> >;
        static auto create(url_view u) -> result<factory>;

        result< producer<T, Ctx> > create_producer() {
            return impl->create_producer().map([&](buffer::s_producer<Ctx> s_prod) {
                return producer<T, Ctx>{ mapper, std::move(s_prod) };
            });
        }

        result< consumer<T, Ctx> > create_consumer() {
            return impl->create_consumer().map([&](buffer::s_consumer<Ctx> s_cons) {
                return consumer<T, Ctx>{ mapper, std::move(s_cons) };
            });
        }

        const sardine::interface::mapping_descriptor& mapping_descriptor() const {
            return mapping_desc;
        }

    };

} // namespace sardine
