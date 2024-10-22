#pragma once

#include <sardine/type.hpp>
#include <sardine/mapping_descriptor/base.hpp>

#include <memory>

namespace sardine::buffer
{

namespace interface
{

    struct view_t
    {
        virtual ~view_t() = default;

        virtual span_b bytes() = 0;
        virtual sardine::url url() const = 0;
    };

    template <typename Ctx>
    struct producer : view_t
    {
        virtual ~producer() = default;

        virtual void   send(Ctx&) = 0;
        virtual void   revert(Ctx&) = 0;

        // virtual sardine::url url() = 0;
    };

    template <typename Ctx>
    struct consumer : view_t
    {
        virtual ~consumer() = default;

        virtual void   recv(Ctx&) = 0;
        virtual void   revert(Ctx&) = 0;

        // virtual sardine::url url() = 0;
    };

} // namespace interface

    using s_view = std::shared_ptr< interface::view_t >;

    template<typename Ctx>
    using s_producer = std::shared_ptr< interface::producer<Ctx> >;

    template<typename Ctx>
    using s_consumer = std::shared_ptr< interface::consumer<Ctx> >;

namespace interface
{

    template<typename Ctx>
    struct factory
    {
        virtual ~factory() = default;

        //Note: factory cannot guarantee that the producer and consumer are always available.
        // Some factories may be read-only, some may be write-only (unlikely).

        virtual result< s_producer<Ctx> > create_producer() = 0;
        virtual result< s_consumer<Ctx> > create_consumer() = 0;

        virtual const sardine::interface::mapping_descriptor& mapping_descriptor() const = 0;
    };

} // namespace interface

    template<typename Ctx>
    using s_factory = std::shared_ptr< interface::factory<Ctx> >;

} // namespace sardine::buffer
