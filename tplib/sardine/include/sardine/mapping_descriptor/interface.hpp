#pragma once

#include <sardine/type.hpp>
#include <sardine/type/url.hpp>

#include <emu/detail/dlpack_types.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace sardine
{

    constexpr auto extent_key = "extent";
    constexpr auto stride_key = "stride";

    // constexpr auto device_type_key = "dt";
    // constexpr auto device_id_key = "di";
    constexpr auto data_type_code_key = "dtc";
    constexpr auto bits_key = "bits";
    constexpr auto lanes_key = "lanes";

    constexpr auto offset_key = "offset";
    constexpr auto is_const_key = "const";

    // using emu::dlpack::device_t;
    using emu::dlpack::data_type_ext_t;

namespace interface
{

    struct mapping_descriptor
    {
        virtual ~mapping_descriptor() = default;

        virtual std::span<const size_t> extents() const = 0;

        virtual bool is_strided() const { return false; }
        virtual std::span<const size_t> strides() const { return {}; }

        // virtual device_t device() const = 0;

        virtual data_type_ext_t data_type() const = 0;

        virtual size_t offset() const = 0;
        virtual bool is_const() const { return false; }

        size_t item_size() const {
            auto dt = data_type();
            return dt.bits * dt.lanes / CHAR_BIT;
        }

    };

} // namespace interface

    inline void update_url(url& u, const interface::mapping_descriptor& desc) {
        auto params = u.params();

        auto dtype = desc.data_type();
        params.set(data_type_code_key, fmt::to_string(dtype.code));
        params.set(bits_key, fmt::to_string(dtype.bits));
        // if dtype is not vectorized, do not bother specifying lanes number.
        if (dtype.lanes != 1)
            params.set(lanes_key, fmt::to_string(dtype.lanes));

        auto extents = desc.extents();

        if (extents.size() > 0) {
            params.set(extent_key, fmt::format("[{}]", fmt::join(extents, ",")));

            if (desc.is_strided())
                params.set(stride_key, fmt::format("[{}]", fmt::join(desc.strides(), ",")));
        }

        if(desc.is_const())
            // set the key with no value.
            params.append({is_const_key, nullptr});

    }


} // namespace sardine
