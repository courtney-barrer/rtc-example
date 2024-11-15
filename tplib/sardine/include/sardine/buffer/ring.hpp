#pragma once

#include <sardine/concepts.hpp>
#include <sardine/context.hpp>
#include <sardine/buffer/base.hpp>
#include <sardine/buffer/impl.hpp>
#include <sardine/mapping_descriptor/base.hpp>
// #include <sardine/buffer/adaptor.hpp>

namespace sardine::ring
{

    constexpr auto url_scheme = "ring";

    constexpr auto data_key       = "r_data";
    constexpr auto index_key      = "r_index";
    constexpr auto offset_key     = "r_offset";
    constexpr auto size_key       = "r_size";
    constexpr auto stride_key     = "r_stride";
    constexpr auto buffer_nb_key  = "r_buffer_nb";
    constexpr auto policy_key     = "r_policy";

// namespace detail
// {

//     template<typename T>
//     struct reduce_view;

//     template<cpts::view V>
//     struct reduce_view<V> {
//         using type = typename V::value_type;
//     };

//     template<cpts::mdview V>
//     struct reduce_view<V> {
//         using type = decltype(emu::submdspan(std::declval<V>(), 0));
//     };

// } // namespace detail

//     template<typename T>
//     using reduce_type = typename detail::reduce_view<T>::type;

    enum class next_policy {
        last, next, check_next
    };

    inline auto format_as(next_policy p) {
        switch (p) {
            case next_policy::last : return "last";
            case next_policy::next : return "next";
            case next_policy::check_next : return "check_next";
        }
        EMU_UNREACHABLE;
    }

    inline result< next_policy > parse_policy( std::string_view s) {
        if (s == "last") return next_policy::last;
        if (s == "next") return next_policy::next;
        if (s == "check_next") return next_policy::check_next;

        return make_unexpected( error::ring_url_invalid_policy );
    }

    struct index
    {

        box<size_t, host_context> global_index; // move to 2 values to be able to detect full vs empty.
        size_t idx;
        size_t buffer_nb;
        next_policy policy;

        size_t save_index = {};

        index(box<size_t, host_context> global_index, size_t buffer_nb, next_policy policy);
        index(size_t& global_index, size_t buffer_nb, next_policy policy);

        void incr_local();
        void decr_local();

        // check if the next index is available
        bool has_next(host_context& ctx);

        void send(host_context& ctx);
        void recv(host_context& ctx);

        void revert_send(host_context& ctx);
        void revert_recv(host_context& ctx);

        sardine::url url() const {
            sardine::url u = global_index.url();

            auto param_ref = u.params();

            param_ref.set(buffer_nb_key, fmt::to_string(buffer_nb));
            param_ref.set(policy_key, fmt::to_string(policy));

            return u;
        }

        static result<index> open(sardine::url url) {
            auto params = url.params();

            auto buffer_nb = urls::try_parse_at<size_t>(params, buffer_nb_key);
            EMU_TRUE_OR_RETURN_UN_EC(buffer_nb, error::ring_url_missing_buffer_nb);

            auto opt_policy = urls::try_get_at(params, policy_key);
            EMU_TRUE_OR_RETURN_UN_EC(opt_policy, error::ring_url_missing_policy);
            auto policy = parse_policy(*move(opt_policy));
            EMU_TRUE_OR_RETURN_ERROR(policy);

            return box<size_t, host_context>::open(url).map([&](auto global_index) {
                return index(global_index, *buffer_nb, *policy);
            });
        }


        // static result<index> open(sardine::url url, size_t buffer_nb, next_policy policy);

        // template<typename Idx>
        // result<index> create(Idx&& idx, size_t buffer_nb, next_policy policy = next_policy::last) {
        //     auto url = url_of(std::forward<Idx>(idx));
        //     if (not url)
        //         return unexpected(url.error());

        //     auto b = box<size_t>::open(*url);
        //     if (not b)
        //         return unexpected(b.error());

        //     return index(*move(b), buffer_nb, policy);
        // }

    };

    struct view_t
    {

        sardine::url data_url;
        span_b data;
        index idx;
        size_t size;   // size of the buffer
        size_t offset; // distance from the start of the buffer
        size_t stride; // distance between buffers
        size_t element_size;

        view_t(sardine::url data_url, span_b data, index idx, size_t size, size_t offset, size_t stride)
            : data_url(std::move(data_url))
            , data(std::move(data))
            , idx(std::move(idx))
            , size(size)
            , offset(offset)
            , stride((stride == std::dynamic_extent) ? size : stride)
        {}

        view_t(const view_t&) = default;

        // view_t(interface_t data, index idx, size_t size, size_t offset = 0, size_t stride = std::dynamic_extent)
        //     : view_t(sardine::view_t<T>(std::move(data)), std::move(idx), size, offset, stride)
        // {}

        byte* data_handle() {
            return data.data();
        }

        span_b bytes() {
            return data.subspan(idx.idx * stride + offset, size);
        }

        sardine::url url() const {
            sardine::url url(fmt::format("{}://", url_scheme));
            auto param_ref = url.params();

            param_ref.set(data_key  , data_url.c_str());
            param_ref.set(index_key , idx.url().c_str());
            param_ref.set(offset_key, fmt::to_string(offset));
            param_ref.set(size_key  , fmt::to_string(size));
            param_ref.set(stride_key, fmt::to_string(stride));

            return url;
        }

        // static result< view_t > create(span_b data, index idx, size_t size, size_t offset = 0, size_t stride = std::dynamic_extent) {
        //     return url_from_bytes(data).map([&](auto url) {
        //         return view_t(url, std::move(data), std::move(idx), size, offset, stride);
        //     });
        // }
    };

    template<typename Ctx>
    struct producer : view_t
    {

        using view_t::view_t;

        producer(view_t view)
            : view_t(std::move(view))
        {
            // producer always set index to the next buffer.
            this->idx.incr_local();
        }

        void send(Ctx& ctx) {
            idx.send(ctx);
        }

        void revert(Ctx& ctx) {
            idx.revert_send(ctx);
        }

    };

    template<typename Ctx>
    struct consumer : view_t
    {

        using view_t::view_t;

        consumer(view_t view)
            : view_t(std::move(view))
        {}

        void recv(Ctx& ctx) {
            idx.recv(ctx);
        }

        void revert(Ctx& ctx) {
            idx.revert_recv(ctx);
        }

    };


    template<typename Ctx>
    struct factory : sardine::buffer::interface::factory<Ctx>
    {
        default_mapping_descriptor mapping_descriptor_;
        view_t view;
        size_t offset_; // should I keep the offset ?

        factory(default_mapping_descriptor mapping_descriptor, url_view u, span_b shm_data, index idx, size_t size, size_t offset, size_t stride)
            : mapping_descriptor_(std::move(mapping_descriptor))
            , view(u, shm_data, std::move(idx), size, offset, stride)
        {}

        const sardine::interface::mapping_descriptor& mapping_descriptor() const {
            return mapping_descriptor_;
        }

        // std::span<const size_t> extents() const override { return mapping_descriptor.extents(); }

        // bool is_strided() const override { return mapping_descriptor.is_strided(); }
        // std::span<const size_t> strides() const override { return mapping_descriptor.strides(); }

        // data_type_t data_type() const override {

        // }

        // size_t offset() const override { return mapping_descriptor.offset(); }
        // bool is_const() const override {

        // }

        result< buffer::s_producer<Ctx> > create_producer() override {
            return buffer::make_s_producer<Ctx>(producer<Ctx>{view});
        }

        result< buffer::s_consumer<Ctx> > create_consumer() override {
            return buffer::make_s_consumer<Ctx>(consumer<Ctx>{view});
        }

        static result< buffer::s_factory<Ctx> > create( url_view u, emu::dlpack::device_type_t requested_dt) {
            auto params = u.params();

            auto size = EMU_UNWRAP_OR_RETURN_UNEXPECTED(urls::try_parse_at<size_t>(params, size_key),
                                                        error::ring_url_missing_size);

            auto offset = urls::try_parse_at<size_t>(params, offset_key).value_or(0);
            auto stride = urls::try_parse_at<size_t>(params, stride_key).value_or(std::dynamic_extent);

            auto data_url = EMU_UNWRAP_OR_RETURN_UNEXPECTED(urls::try_get_at(params, data_key).map([](auto value){ return url(value); }),
                                                            error::ring_url_missing_data);

            auto idx_url = EMU_UNWRAP_OR_RETURN_UNEXPECTED(urls::try_get_at(params, index_key),
                                                           error::ring_url_missing_index);

            auto memory_h = EMU_UNWRAP(detail::bytes_from_url(data_url, requested_dt) );

            auto mapping_desc = EMU_UNWRAP(make_mapping_descriptor(params));

            return index::open(url(idx_url)).map([&](auto idx) {
                return std::make_shared<factory>(std::move(mapping_desc), u, memory_h, std::move(idx), size, offset, stride);
            });

        }
    };


namespace detail
{

    template<typename T>
        requires cpts::closable_lead_dim< sardine::mapper< emu::decay<T> > >
    auto make_view(T&& data, index idx, size_t offset = 0)
    {
        auto mapper = mapper_from(EMU_FWD(data));

        auto view = sardine::as_bytes(data);

        auto closed_accessor = mapper.close_lead_dim();

        offset += closed_accessor.offset();

        constexpr static size_t t_size = sizeof(typename decltype(closed_accessor)::value_type);

        constexpr static auto requested_device_type = emu::location_type_of<emu::decay<T>>::device_type;

        return sardine::detail::url_from_bytes(view, requested_device_type).map([&](auto url) {
            return std::make_pair(
                std::move(closed_accessor),
                view_t(
                    std::move(url), view,
                    std::move(idx),
                    closed_accessor.size() * t_size,
                    offset * t_size,
                    mapper.lead_stride() * t_size
                )
            );
        });

    }

} // namespace detail

    template<typename T>
    auto make_view(T&& data, index idx, size_t offset = 0) {
        return detail::make_view(EMU_FWD(data), std::move(idx), offset).map([&](auto pair) {
            auto [accessor, view] = std::move(pair);
            using data_view_t = typename decltype(accessor)::type;
            return sardine::view_t<data_view_t>(accessor, buffer::make_s_view(std::move(view)));
        });
    }


    template<typename Ctx, typename T>
    auto make_producer(T&& data, index idx, size_t offset = 0) {
        return detail::make_view(EMU_FWD(data), std::move(idx), offset).map([&](auto pair) {
            auto [accessor, view] = std::move(pair);
            using data_view_t = typename decltype(accessor)::view_type;
            return sardine::producer<data_view_t, Ctx>(accessor, buffer::make_s_producer<Ctx>(producer<Ctx>(std::move(view))));
        });
    }

    template<typename Ctx, typename T>
    auto make_consumer(T&& data, index idx, size_t offset = 0) {
        return detail::make_view(EMU_FWD(data), std::move(idx), offset).map([&](auto pair) {
            auto [accessor, view] = std::move(pair);
            using data_view_t = typename decltype(accessor)::view_type;
            return sardine::consumer<data_view_t, Ctx>(accessor, buffer::make_s_consumer<Ctx>(consumer<Ctx>(std::move(view))));
        });
    }

} // namespace sardine::ring
