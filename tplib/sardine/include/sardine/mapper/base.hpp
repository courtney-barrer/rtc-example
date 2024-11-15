#pragma once

// #include <sardine/type.hpp>
// #include <sardine/type/url.hpp>
// #include <sardine/error.hpp>

// #include <sardine/mapping/base.hpp>

// #include <emu/concepts.hpp>
// #include <emu/span.hpp>
// #include <emu/mdspan.hpp>
// #include <type_traits>


#include <sardine/type.hpp>
#include <sardine/error.hpp>
#include <sardine/mapping_descriptor.hpp>
#include <sardine/sink.hpp>

#include <emu/type_traits.hpp>
#include <emu/concepts.hpp>
#include <emu/cstring_view.hpp>
#include <emu/capsule.hpp>
#include <emu/numeric_type.hpp>
#include <emu/type_name.hpp>

#include <ranges>

namespace sardine
{

namespace mapper_cpts
{

    // A view is a non owning type that wrap over a pointer. It includes C++ reference and pointers.
    template<typename T>
    concept view = (emu::cpts::pointer<T> or emu::cpts::ref<T>)
        or (emu::cpts::any_view<T> and not emu::cpts::any_owning<T>)
        or (emu::cpts::any_string_view<T>);

    // Similar to a view, it also accept a emu::capsule that allow to extend value lifetime.
    // Only supported by emu library, under the any_owning concept.
    template<typename T>
    concept capsule_container = emu::cpts::any_owning<T>;

    // A copy container is a owning type that cannot old any external pointer.
    // It includes regular POD types, vector and string.
    template<typename T>
    concept copy_container = (not view<T> and not capsule_container<T>);

    template<typename T>
    concept contiguous = std::ranges::contiguous_range<T>;

    // Includes all types that use mapping to access data. For now, only md[span/container] model this concept.
    template<typename T>
    concept mapped = emu::cpts::any_mdspan<T>;

    // Concept that describe a single value type.
    template<typename T>
    concept scalar = (not contiguous<T>) and (not mapped<T>);

} // namespace mapper_cpts

    template<typename T>
    struct mapper
    {
        using view_type = T&;
        using type = T;
        // using convert_type = T&;
        using value_type = emu::rm_ref<T>; // just in case
        // using container_type = T;

        size_t offset_;

        emu::capsule capsule;

        static result< mapper > from_mapping_descriptor(const interface::mapping_descriptor& md, emu::capsule&& capsule) {
            EMU_TRUE_OR_RETURN_UN_EC_LOG(md.item_size() == sizeof(value_type), error::mapper_item_size_mismatch,
                "incompatible item size between mapping : {} and requested type {} : {}", md.item_size(), emu::type_name<value_type>, sizeof(value_type));
            EMU_TRUE_OR_RETURN_UN_EC(md.extents().size() == 0            , error::mapper_not_scalar);
            EMU_TRUE_OR_RETURN_UN_EC(emu::is_const<T> or (not md.is_const()), error::mapper_const);

            return mapper{md.offset(), std::move(capsule)};
        }

        static mapper from(const T&) {
            //TODO: consider capturing T in a capsule. NO !
            return mapper{0, emu::capsule()};
        }

        size_t size() const { return 1; }
        size_t offset() const { return offset_; }
        // size_t lead_stride() const { return 1; }

        T& convert(span_b buffer) const & {
            //Note: We do nothing with the capsule here, since we don't have "scalar" types that takes it.
            EMU_ASSERT_MSG(buffer.size() >= sizeof(value_type), "Buffer size is invalid.");
            return *(reinterpret_cast<value_type*>(buffer.data()) + offset_);
        }

        T& convert(span_b buffer) && {
            // Mapper will be destroyed after this call, so we need to keep the capsule alive.
            sink(std::move(capsule));

            return static_cast<const mapper*>(this)->convert(buffer);
        }

        template<typename TT>
        static auto as_bytes(TT&& value) {
            return emu::as_auto_bytes(std::span{&value, 1});
        }

        default_mapping_descriptor mapping_descriptor() const {
            return default_mapping_descriptor(
                  /* extents = */ std::vector<size_t>{},
                  /* strides = */ std::vector<size_t>{},
                  /* data_type = */ emu::dlpack::data_type_ext<T>,
                  /* offset = */ offset_,
                  /* is_const = */ emu::is_const<T>
            );
        }

    };

    template<mapper_cpts::contiguous T>
    struct mapper< T >
    {
        using view_type = T;
        using type = T;
        // only consistent way to know if range element is const or not.
        using element_type = emu::rm_ref<std::ranges::range_reference_t<T>>;
        using value_type = emu::rm_const<element_type>;

        /**
         * @brief This fake span is used to keep track of the offset and size
         *
         * It should never be used to access the data, only to compute the subspan ibn the convert method.
         *
         */
        // fake_span_t fake_span;
        size_t offset_;
        size_t size_;

        emu::capsule capsule;

    public:
        static result< mapper > from_mapping_descriptor(const interface::mapping_descriptor& md, emu::capsule capsule) {

            EMU_TRUE_OR_RETURN_UN_EC_LOG(md.item_size() == sizeof(value_type), error::mapper_item_size_mismatch,
                "incompatible item size between mapping : {} and requested type {} : {}", md.item_size(), emu::type_name<value_type>, sizeof(value_type));

            EMU_TRUE_OR_RETURN_UN_EC_LOG(not md.is_strided(), error::mapper_incompatible_stride,
                "mapping is strided, mapper does not support it. strides are {}", md.strides());

            EMU_TRUE_OR_RETURN_UN_EC_LOG(emu::is_const<T> or (not md.is_const()), error::mapper_const,
                "Mapping to const data, but mapper requires mutable data type {}", emu::type_name<element_type>);

            // We accept n dimension mapping as long as it is contiguous.
            size_t size = 1; for (auto e : md.extents()) size *= e;

            return mapper{md.offset(), size, capsule};
        }

        static mapper from(const T& value) {
            return mapper{0, std::ranges::size(value), emu::capsule()};
        }

        size_t size() const { return size_; }
        size_t offset() const { return offset_; }
        size_t lead_stride() const { return 1; }

        T convert(span_b buffer) const & {
            EMU_ASSERT_MSG(buffer.size() >= size() * sizeof(value_type), "Buffer size is invalid.");

            auto view = emu::as_t<element_type>(buffer).subspan(offset_, size_);

            if constexpr (mapper_cpts::capsule_container<T>)
                return T(view.begin(), view.end(), capsule);
            else
                return T(view.begin(), view.end());
        }

        T convert(span_b buffer) && {
            // Mapper will be destroyed after this call, so we need to keep the capsule alive.
            if constexpr (not mapper_cpts::capsule_container<T>)
                sink(std::move(capsule));

            return static_cast<const mapper*>(this)->convert(buffer);
        }

        template<typename TT>
        static auto as_bytes(TT&& value) {
            return emu::as_auto_bytes(std::span{value});
        }

        mapper<element_type> close_lead_dim() const {
            return {offset()};
        }

        default_mapping_descriptor mapping_descriptor() const {
            return default_mapping_descriptor(
                  /* extents = */ std::vector<size_t>{size_},
                  /* strides = */ std::vector<size_t>{},
                  /* data_type = */ emu::dlpack::data_type_ext<value_type>,
                  /* offset = */ offset_,
                  /* is_const = */ emu::is_const<element_type>
            );
        }

        mapper subspan(size_t new_offset, size_t new_size) const {
            EMU_ASSERT_MSG(new_offset + new_size <= size_, "Subspan is out of bound.");

            std::span<const byte> fake_span{reinterpret_cast<const byte*>(offset_), size_};

            // let span do the computation to get the new subspan.
            auto sv = fake_span.subspan(new_offset, new_size);

            // sv.data() is the new offset and sv.size() is the new size.
            return {reinterpret_cast<size_t>(sv.data()), sv.size(), capsule};
        }
    };


    template<mapper_cpts::mapped T>
    struct mapper< T >
    {
        using view_type = T;
        using type = T;

        using element_type = typename T::element_type;

        using value_type = typename T::value_type;
        using extents_type = typename T::extents_type;
        using layout_type      = typename T::layout_type;
        using accessor_type    = typename T::accessor_type;

        using mapping_type = typename T::mapping_type;
        // using container_type = emu::mdcontainer< value_type, extents_type, layout_type, accessor_type >;


        size_t offset_;
        mapping_type mapping_;

        emu::capsule capsule;

        static result< mapper > from_mapping_descriptor(const interface::mapping_descriptor& md, emu::capsule capsule) {
            EMU_TRUE_OR_RETURN_UN_EC_LOG(md.item_size() == sizeof(value_type), error::mapper_item_size_mismatch,
                "incompatible item size between mapping : {} and requested type {} : {}", md.item_size(), emu::type_name<value_type>, sizeof(value_type));

            EMU_TRUE_OR_RETURN_UN_EC(emu::is_const<T> or (not md.is_const()), error::mapper_const);

            return create_mapping<mapping_type>(md).map([&](auto mapping) {
                return mapper{md.offset(), mapping, capsule};
            });
        }

        static mapper from(const T& mdspan) {
            return mapper{0, mdspan.mapping()};
        }

        size_t size() const { return mapping_.required_span_size(); }
        size_t offset() const { return offset_; }
        size_t lead_stride() const { return mapping_.stride(0); } // what about layout_f ?

        auto close_lead_dim() const {
            return submdspan(0);
        }

        T convert(span_b buffer) const & {
            auto view = emu::as_t<element_type>(buffer).subspan(offset_);

            EMU_ASSERT_MSG(view.size() >= mapping_.required_span_size(), "Buffer size is invalid.");

            if constexpr (mapper_cpts::capsule_container<T>)
                return T(view.data(), mapping_, capsule);
            else
                return T(view.data(), mapping_);
        }

        T convert(span_b buffer) && {
            // Mapper will be destroyed after this call, so we need to keep the capsule alive.
            if constexpr (not mapper_cpts::capsule_container<T>)
                sink(std::move(capsule));

            return static_cast<const mapper*>(this)->convert(buffer);
        }

        template<typename TT>
        static auto as_bytes(TT&& value) {
            return emu::as_auto_bytes(std::span{value.data_handle(), value.mapping().required_span_size()});
        }

        default_mapping_descriptor mapping_descriptor() const {
            std::vector<size_t> extents; extents.resize(extents_type::rank());
            for (size_t i = 0; i < extents_type::rank(); ++i)
                extents[i] = mapping_.extents().extent(i);

            std::vector<size_t> strides;
            if constexpr (not mapping_type::is_always_exhaustive())
                if (not mapping_.is_exhaustive()) {
                    strides.resize(extents_type::rank());
                    for (size_t i = 0; i < extents_type::rank(); ++i)
                        strides[i] = mapping_.stride(i);
                }


            return default_mapping_descriptor(
                  /* extents = */ std::move(extents),
                  /* strides = */ std::move(strides),
                  /* data_type = */ emu::dlpack::data_type_ext<value_type>,
                  /* offset = */ offset_,
                  /* is_const = */ emu::is_const<element_type>
            );
        }

        template<class... SliceSpecifiers>
        auto submdspan(SliceSpecifiers... specs) const
        {

            // Create a fake mdspan from offset and size. Use deduction guide.
            // considere the offset as a pointer to the first element of the mdspan.
            // byte is 1 byte long, so the offset is in bytes.
            emu::mdspan fake_mdspan(reinterpret_cast< const byte* >(offset_), this->mapping_);


            // let mdspan do the computation to get the new submdspan.
            auto sv = emu::submdspan(fake_mdspan, specs...);

            using new_mapping_t = typename decltype(sv)::mapping_type;
            using new_mdspan_t = emu::mdspan<element_type, typename new_mapping_t::extents_type, typename new_mapping_t::layout_type>;

            // The pointer returned by data_handle is the new offset.
            return sardine::mapper<new_mdspan_t>{reinterpret_cast<size_t>(sv.data_handle()), sv.mapping(), capsule};
        }

    };

} // namespace sardine
