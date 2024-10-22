#pragma once

#include <sardine/type.hpp>
#include <sardine/type/url.hpp>
#include <sardine/error.hpp>

#include <sardine/mapping_descriptor/interface.hpp>

#include <array>
#include <vector>
#include <ranges>
#include <concepts>

namespace sardine
{

    struct default_mapping_descriptor : interface::mapping_descriptor
    {
        std::vector<size_t> extents_;
        std::vector<size_t> strides_;

        // device_t device_;
        data_type_ext_t data_type_;

        size_t offset_;
        bool is_const_;

        default_mapping_descriptor() = default;

        default_mapping_descriptor(
            std::vector<size_t> extents, std::vector<size_t> strides,
            // device_t device,
            data_type_ext_t data_type,
            size_t offset, bool is_const
        )
            : extents_(std::move(extents))
            , strides_(std::move(strides))
            // , device_(device)
            , data_type_(data_type)
            , offset_(offset)
            , is_const_(is_const)
        {}

        default_mapping_descriptor(const default_mapping_descriptor&) = default;

        default_mapping_descriptor& operator=(const default_mapping_descriptor&) = default;

        ~default_mapping_descriptor() = default;

        std::span<const size_t> extents() const override { return extents_; }
        bool is_strided() const override { return not strides_.empty(); }
        std::span<const size_t> strides() const override { return strides_; }

        // device_t device() const override { return device_; }
        data_type_ext_t data_type() const override { return data_type_; }

        size_t offset() const override { return offset_; }
        bool is_const() const override { return is_const_; }
    };

    // consider returning `std::unique_ptr<interface::mapping_descriptor>`
    inline result<default_mapping_descriptor> make_mapping_descriptor(urls::params_view params) {
        using vector_t = std::vector<size_t>;

        auto extents = urls::try_parse_at<vector_t>(params, extent_key).value_or(vector_t{});

        auto strides = urls::try_parse_at<vector_t>(params, stride_key).value_or(vector_t{});

        using namespace emu::dlpack;

        uint8_t code = EMU_UNWRAP(urls::try_parse_number_at<uint8_t>(params, data_type_code_key));
        uint64_t bits = EMU_UNWRAP(urls::try_parse_number_at<uint64_t>(params, bits_key));
        uint16_t lanes = EMU_UNWRAP(urls::try_parse_number_at(params, lanes_key, uint16_t(1)));

        data_type_ext_t data_type{code, bits, lanes};

        auto offset = urls::try_parse_at<size_t>(params, offset_key).value_or(0);
        auto is_const = urls::try_parse_at<bool>(params, is_const_key).value_or(false);

        return {in_place, std::move(extents), std::move(strides)/* , device*/, data_type, offset, is_const};
    }

    template<typename Mapping>
    auto create_mapping(const interface::mapping_descriptor& descriptor) -> result< Mapping > {
        using Extent = typename Mapping::extents_type;
        using Layout = typename Mapping::layout_type;

        using array_t = std::array<size_t, Extent::rank()>;

        array_t extents; std::ranges::copy(descriptor.extents(), extents.begin());

        constexpr static auto rank = Mapping::extents_type::rank();

        if (extents.size() != rank)
            return make_unexpected(error::mapper_rank_mismatch);

        if constexpr ( std::same_as<Layout, emu::layout_right>
                    or std::same_as<Layout, emu::layout_left> ) {
            return Mapping(extents);

        } else if constexpr (std::same_as<Layout, emu::layout_stride> ) {

            if (not descriptor.is_strided())
                // not having stride if fine, we can compute it.
                return Mapping(extents);
            else {
                array_t strides; std::ranges::copy(descriptor.strides(), strides.begin());

                if (strides.size() != rank)
                    return make_unexpected(error::mapper_rank_mismatch);

                return Mapping(extents, strides);
            }

        } else
            static_assert(emu::dependent_false<Mapping>, "Layout is not supported.");
    }

} // namespace sardine
