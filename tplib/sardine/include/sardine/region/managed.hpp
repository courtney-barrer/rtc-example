#pragma once

#include <memory>
#include <sardine/type.hpp>
#include <sardine/type/url.hpp>
#include <sardine/region/managed/utility.hpp>

#include <emu/utility.hpp>
#include <emu/optional.hpp>
#include <emu/span.hpp>
#include <emu/cstring_view.hpp>

#include <fmt/format.h>

#include <future>
#include <string>
#include <stdexcept>

namespace sardine::region
{

namespace managed::spe
{

    template<typename T>
    struct managed_adaptor;

} // namespace managed::spe

    struct managed_t
    {
        using char_ptr_holder_t = managed::char_ptr_holder_t;

        managed::shared_memory* shm_;

        managed_t(managed::shared_memory* shm)
            : shm_{shm}
        {}

        template<typename T, typename... Args>
            requires (not emu::is_ref<T>)
        auto force_create(char_ptr_holder_t name, Args&&... args) -> decltype(auto) {
            return atomic_call([&] () -> decltype(auto) {
                destroy<T>(name); // no need to check if exist, destroy will do nothing if not exist.
                return create_unchecked<T>(name, EMU_FWD(args)...);
            });
        }

        template<typename T, typename... Args>
            requires (not emu::is_ref<T>)
        auto create(char_ptr_holder_t name, Args&&... args) -> decltype(auto) {
            return atomic_call([&] () -> decltype(auto) {
                if (!exist<T>(name))
                    return create_unchecked<T>(name, EMU_FWD(args)...);
                else
                    throw std::runtime_error(fmt::format("{} {} already exist", emu::type_name<T>, name));
            });
        }

        template<typename T>
        auto open(char_ptr_holder_t name) -> decltype(auto) {
            return atomic_call([&] () -> decltype(auto) {
                if (exist<T>(name))
                    return open_unchecked<T>(name);
                else
                    throw std::runtime_error(fmt::format("{} {} not found", emu::type_name<T>, name));
            });
        }

        template<typename T, typename... Args>
            requires (not emu::is_ref<T>)
        auto open_or_create(char_ptr_holder_t name, Args&&... args) -> decltype(auto) {
            return atomic_call([&] () -> decltype(auto) {
                if (exist<T>(name))
                    return open_unchecked<T>(name);
                else
                    return create_unchecked<T>(name, EMU_FWD(args)...);
            });
        }

        template<typename T>
        void set(char_ptr_holder_t name, T&& value) {
            open<emu::decay<T>>(name) = EMU_FWD(value);
        }

        template<typename T>
        bool type_check(char_ptr_holder_t name) {
            // TODO: check named_range to see if nothing exist for real and detect type mismatch.
            // HOW ??? Damn you past julien...
            return true;
        }

        template<typename T>
            requires (not emu::is_ref<T>)
        bool exist(char_ptr_holder_t name) {
            return managed::spe::managed_adaptor<T>::exist(*this, name);
        }

        template<typename T>
            requires (not emu::is_ref<T>)
        bool destroy(char_ptr_holder_t name) {
            return managed::spe::managed_adaptor<T>::destroy(*this, name);
        }

        template<typename T>
            requires (not emu::is_ref<T>)
        auto offset_of(const T& value) const -> decltype(auto) {
            auto addr = managed::spe::managed_adaptor<T>::address_of(*this, value);
            return shm().get_handle_from_address(addr);
        }

        template<typename T>
            requires (not emu::is_ref<T>)
        auto from_offset(managed::handle_t offset) const -> decltype(auto) {
            std::byte* address = reinterpret_cast<std::byte*>(shm_->get_address_from_handle(offset));
            return managed::spe::managed_adaptor<T>::from_region(*this, span_b{address, 1});
        }

        template<typename T, typename... Args>
        auto create_unchecked(char_ptr_holder_t name, Args&&... args) -> decltype(auto) {
            return managed::spe::managed_adaptor<T>::create(*this, name, EMU_FWD(args)...);
        }

        template<typename T>
        auto open_unchecked(char_ptr_holder_t name) -> decltype(auto) {
            auto& seg = segment_manager();
            auto [address, count] = seg.find<std::byte>(name);
            return managed::spe::managed_adaptor<T>::from_region(*this, span_b{reinterpret_cast<std::byte*>(address), count});
        }

        managed::shared_memory& shm() const;

        managed::segment_manager& segment_manager() const;

        managed::named_range named() const;

    private:

        template<typename Fn>
        auto atomic_call(Fn fn) -> decltype(auto) {
            using ret_type = decltype(fn());

            // Would have be great to use std::optional but does not allow reference.
            std::promise<ret_type> p;

            // Cannot retrieve the return value from atomic_func, so we use a promise to pass back the value.
            // atomic_func only takes function reference, function need to be a lvalue.
            auto l = [&]{ p.set_value(fn()); };
            shm().atomic_func(l);

            auto f = p.get_future();
            return f.get();
        }

    };

namespace managed
{

namespace spe
{
    template<typename T>
    struct default_managed_adaptor
    {

        template<typename... Args>
        static auto create(managed_t shm, char_ptr_holder_t name, Args&&... args) -> decltype(auto) {
            auto& seg = shm.segment_manager();
            allocator<std::byte> alloc(&seg);

            return std::apply([&](auto&&... args) -> decltype(auto) {
                return *seg.construct<T>(name)(EMU_FWD(args)...);
            }, std::uses_allocator_construction_args<T>(alloc, EMU_FWD(args)...));
        }

        static auto exist(managed_t shm, char_ptr_holder_t name) -> bool {
            auto& seg = shm.segment_manager();
            return seg.find<T>(name).first != 0;
        }

        static auto destroy(managed_t shm, char_ptr_holder_t name) -> bool {
            auto& seg = shm.segment_manager();
            return seg.destroy<T>(name);
        }

        static auto from_region(managed_t, span_b region) -> T&/* decltype(auto) */ {
            EMU_ASSERT_MSG(region.size() >= sizeof(T), "Region is too small to contain a value of type {}");
            //TODO: test alignment ?

            return *reinterpret_cast<T*>(region.data());
        }

        static auto region_of(const T& value) -> span_cb {
            return std::as_bytes(std::span{&value, 1});
        }

    };

    template< typename Span >
        requires (emu::cpts::span<Span> or emu::host::cpts::span<Span> )
    struct default_managed_adaptor< Span >
    {
        // T may be const, span::value_type is not.
        using value_type = typename Span::value_type;

        template<typename... Args>
        static auto create(managed_t shm, char_ptr_holder_t name, size_t count, Args&&... args) {
            auto& seg = shm.segment_manager();
            allocator<std::byte> alloc(&seg);

            //TODO: delegate create to managed_adaptor<value_type>

            return std::apply([&](auto&&... args) {
                return Span{seg.construct<value_type>(name)[count](EMU_FWD(args)...), count};
            }, std::uses_allocator_construction_args<value_type>(alloc, EMU_FWD(args)...));
        }

        static auto exist(managed_t shm, char_ptr_holder_t name) -> bool {
            auto& seg = shm.segment_manager();
            return seg.find<value_type>(name).first != 0;
        }

        static auto destroy(managed_t shm, char_ptr_holder_t name) -> bool {
            auto& seg = shm.segment_manager();
            return seg.destroy<value_type>(name);
        }

        static auto from_region(managed_t, span_b region) -> Span {
            return emu::as_t<value_type>(region);
        }

        static auto region_of(Span value) -> span_cb {
            return std::as_bytes(value);
        }

    };

    template<emu::cpts::any_string_view StringView>
    struct default_managed_adaptor< StringView > : default_managed_adaptor< std::span<typename StringView::value_type> >
    {
        using base_t = default_managed_adaptor< std::span<typename StringView::value_type> >;

        using char_t = typename StringView::value_type;
        using traits_type = typename StringView::traits_type;

        static auto create(managed_t shm, char_ptr_holder_t name, const char_t* str) {
            auto size = traits_type::length(str);

            auto& seg = shm.segment_manager();
            auto span_of_char = std::span{seg.construct<char_t>(name)[size](), size};

            traits_type::copy(span_of_char.data(), str, size);

            return StringView(span_of_char.begin(), span_of_char.end() );
        }

        // template<typename... Args>
        // static auto create(managed_t shm, char_ptr_holder_t name, Args&&... args) {
        //     auto& seg = shm.segment_manager();
        //     allocator<std::byte> alloc(&seg);

        //     auto& str = std::apply([&](auto&&... args) -> decltype(auto) {
        //         return *seg.construct<string_equivalent>(name)(EMU_FWD(args)...);
        //     }, std::uses_allocator_construction_args<T>(alloc, EMU_FWD(args)...));

        //     return StringView( str );
        // }

        static auto from_region(managed_t m, span_b region) -> StringView {
            auto span_of_char = base_t::from_region(m, region);

            //TODO: replace by range constructor when C++23
            return StringView(span_of_char.begin(), span_of_char.end() );
        }

        static auto region_of(const StringView& value) -> span_cb {
            return base_t::region_of(std::span(value)); // conversion from string_view to span.
        }

    };

    template<typename T>
    struct managed_adaptor : default_managed_adaptor<T>
    {};


} // namespace spe

    struct shm_handle {
        emu::cstring_view name;
        managed_t shm;
        handle_t offset;
    };

    managed_t open(std::string name);

    managed_t create(std::string name, size_t file_size);

    managed_t open_or_create(std::string name, size_t file_size);

    optional<shm_handle> find_handle(const std::byte* ptr);

    struct managed_area {
        managed_t shm;
        span_b region;
    };

    constexpr auto url_scheme = "managed";

    optional<result<url>> url_from_bytes(span_cb data);

    result<bytes_and_device> bytes_from_url(url_view u);


    // template<typename T>
    // using from_region_type = std::invoke_result_t<decltype(managed::spe::managed_adaptor<T>::from_region), managed_t, span_b>;

    // template<typename T>
    // constexpr auto element_size(size_t region_size) -> size_t {
    //     if constexpr (emu::cpts::view<T>) {
    //         return sizeof(typename T::element_type);
    //     } else {
    //         return region_size;
    //     }
    // }

    // template<typename T>
    // constexpr decltype(auto) region_view(const T& value) {
    //     if constexpr (emu::cpts::mdspan<T>) {
    //         return std::span(value.data_handle(), value.mapping().required_span_size());
    //     } else {
    //         return value;
    //     }
    // }

    // template<typename T>
    // optional<url> url_of(const T& value) {
    //     // region_view will convert mdspan to span and forward the value otherwise.
    //     // then we get the region of the value.
    //     auto region = spe::managed_adaptor<T>::region_of(region_view(value));

    //     return detail::url_of(region, element_size<T>(region.size())).map([&](auto url) {
    //         // update the url with the value shape and strides.
    //         update_url(url, value);

    //         return url;
    //     });
    // }

    // template<typename T>
    // concept can_handle_type = std::constructible_from<T, from_region_type<std::decay_t<T>>>
    //                        or emu::cpts::span<T>
    //                        or emu::cpts::mdspan<T>;

    // auto view_of(url_view u) -> optional<span_b>;

    // template<typename T>
    // auto from_url(url_view url) -> result<return_type<T>> {
    //     static_assert(can_handle_type<T>, "Invalid type T. Cannot be constructed from managed.");

    //     auto ma = detail::from_url(url);
    //     if (not ma) return unexpected(ma.error());

    //     auto [shm, span, element_size] = *ma;

    //     // if (element_size != element_size_of<T>)
    //     //     return unexpected( fmt::format("Element size mismatch: {} != {}",
    //     //             element_size, element_size_of<T>
    //     //     ));

    //     // shm is only used for non view types.
    //     if constexpr (emu::cpts::mdspan<T> or emu::cpts::span<T>) {
    //         auto j = ::sardine::json::from_url_param(url.params());

    //         return make_accessor<T>(j).map([&](auto as) { return as.convert(span); });
    //     } else
    //         return managed::spe::managed_adaptor<std::decay_t<T>>::from_region(shm, span);

    // }


} // namespace managed

} // namespace sardine::region
