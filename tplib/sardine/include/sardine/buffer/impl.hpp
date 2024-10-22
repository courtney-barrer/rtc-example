#pragma once

#include <sardine/buffer/interface.hpp>
#include <sardine/type/url.hpp>
#include <sardine/mapping_descriptor/base.hpp>
#include <sardine/url.hpp>

#include <emu/macro.hpp>
#include <emu/mdspan.hpp>
#include <emu/info.hpp>

#include <fmt/format.h>

namespace sardine::buffer
{


    /**
     * @brief A basic producer/consumer that uses bytes range as storage.
     *
     * Since producer and consumer are the same, we use the same struct for both.
     *
     */
    template<typename Ctx>
    struct default_producer_consumer
    {
        span_b shm_data;
        sardine::url u;

        default_producer_consumer(span_b data, sardine::url u)
            : shm_data(data)
            , u(move(u))
        {}

        byte* data_handle() {
            return shm_data.data();
        }

        span_b bytes() {
            return shm_data;
        }

        void recv(Ctx&) {}
        void send(Ctx&) {}
        void revert(Ctx&) {}

        url_view url() const { return u; }
    };

namespace native
{

    /**
     * native types provides a way to create a view, producer or consumer from a span of bytes.
     * It does not have any logic, it just wraps the span of bytes.
     */

    struct view_t {
        span_b view_;

        byte* data_handle() { return view_.data(); }
        span_b bytes() { return view_; }

        sardine::url url() const {
            auto maybe_url = sardine::detail::url_from_bytes(view_);

            EMU_UNWRAP_RETURN_IF_TRUE(maybe_url);

            throw std::runtime_error(fmt::format("url_from_bytes failed: {}", maybe_url.error().message()));
        }
    };

    template <typename Ctx>
    struct producer : view_t {

        void send  ( Ctx& ) { }
        void revert( Ctx& ) { }
    };

    template <typename Ctx>
    struct consumer : view_t {

        void recv  ( Ctx& ) { }
        void revert( Ctx& ) { }
    };

} // namespace native

namespace impl
{

    /**
     * @brief
     *
     * @tparam Impl
     */
    template<typename Impl>
    struct view_t : interface::view_t
    {
        Impl impl;

        view_t(Impl impl)
            : impl(std::move(impl))
        {}

        span_b bytes() override { return impl.bytes(); }

        sardine::url url() const override { return impl.url(); }
    };

    template <typename Ctx, typename Impl>
    struct producer : interface::producer<Ctx>
    {
        Impl impl;

        producer(Impl impl)
            : impl(std::move(impl))
        {}

        void send(Ctx& ctx) override { impl.send(ctx);     }
        void revert(Ctx& ctx) override { impl.revert(ctx);   }

        span_b bytes() override { return impl.bytes(); }

        sardine::url url( ) const override { return impl.url();  }
    };

    template <typename Ctx, typename Impl>
    struct consumer : interface::consumer<Ctx>
    {
        Impl impl;

        consumer(Impl impl)
            : impl(std::move(impl))
        {}

        void recv  (Ctx& ctx) override { impl.recv(ctx);     }
        void revert(Ctx& ctx) override { impl.revert(ctx);   }

        span_b bytes() override { return impl.bytes(); }

        sardine::url url( ) const override { return impl.url();  }
    };

} // namespace impl

    /**
     * Instead of inheriting from the interface, simply use the make_s_* functions.
     * It will automatically create the correct type that inherits from the interface.
     */

    template <typename Impl>
    s_view make_s_view(Impl impl) {
        return std::make_shared< impl::view_t<Impl> >(std::move(impl));
    }

    template <typename Ctx, typename Impl>
    s_producer< Ctx > make_s_producer(Impl impl) {
        return std::make_shared< impl::producer<Ctx, Impl> >(std::move(impl));
    }

    template <typename Ctx, typename Impl>
    s_consumer< Ctx > make_s_consumer(Impl impl) {
        return std::make_shared< impl::consumer<Ctx, Impl> >(std::move(impl));
    }

    template<typename T, typename Ctx>
    struct bytes_factory : interface::factory<Ctx>
    {
        span_b shm_data;
        default_mapping_descriptor mapping_descriptor_;
        url u;

        bytes_factory(span_b shm_data, default_mapping_descriptor mapping_descriptor, url_view u)
            : shm_data(shm_data)
            , mapping_descriptor_(move(mapping_descriptor))
            , u(u)
        {}

        const sardine::interface::mapping_descriptor& mapping_descriptor() const {
            return mapping_descriptor_;
        }

        // size_t                  offset()     const override { return mapping_descriptor.offset();     }
        // size_t                  item_size()  const override { return mapping_descriptor.item_size();  }
        // std::span<const size_t> extents()    const override { return mapping_descriptor.extents();    }
        // bool                    is_strided() const override { return mapping_descriptor.is_strided(); }
        // std::span<const size_t> strides()    const override { return mapping_descriptor.strides();    }

        result< buffer::s_producer<Ctx> > create_producer() override {
            return buffer::make_s_producer<Ctx>(buffer::default_producer_consumer<Ctx>{shm_data, u});
        }

        result< buffer::s_consumer<Ctx> > create_consumer() override {
            return buffer::make_s_consumer<Ctx>(buffer::default_producer_consumer<Ctx>{shm_data, u});
        }

        static result< buffer::s_factory<Ctx> > create(url_view u) {
            auto memory_h = sardine::detail::bytes_from_url<T>(u);
            EMU_TRUE_OR_RETURN_ERROR(memory_h);

            auto md = make_mapping_descriptor(u.params());
            EMU_TRUE_OR_RETURN_ERROR(md);

            return std::make_shared<bytes_factory>(*memory_h, *move(md), u);
        }
    };

} // namespace sardine::buffer
