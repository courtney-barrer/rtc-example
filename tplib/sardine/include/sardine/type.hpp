#pragma once

#include <sardine/error.hpp>

#include <emu/type_traits.hpp>
#include <emu/expected.hpp>
#include <emu/optional.hpp>
#include <emu/cstring_view.hpp>
#include <emu/span.hpp>
#include <emu/container.hpp>
#include <emu/detail/dlpack_types.hpp>

#include <boost/system.hpp>
#include <boost/interprocess/creation_tags.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace sardine
{
    using std::size_t, std::move, std::byte, std::string, std::string_view, std::dynamic_extent;

    using emu::span, emu::cstring_view;

    using span_b = std::span<std::byte>;
    using span_cb = std::span<const std::byte>;

    using container_b = emu::container<byte>;
    using container_cb = emu::container<const byte>;

    using emu::optional;

    using emu::nullopt;

    using emu::in_place;

    using emu::dlpack::device_type_t;

    using tl::unexpected;

    using boost::interprocess::create_only_t;
    using boost::interprocess::open_only_t;
    using boost::interprocess::open_read_only_t;
    using boost::interprocess::open_or_create_t;
    using boost::interprocess::open_copy_on_write_t;

    using boost::interprocess::create_only;
    using boost::interprocess::open_only;
    using boost::interprocess::open_read_only;
    using boost::interprocess::open_or_create;
    using boost::interprocess::open_copy_on_write;

    // struct heap_allocator_tag {};

    // constexpr heap_allocator_tag heap_allocator;

    struct bytes_and_device {
        span_b region;
        span_b data;
        emu::dlpack::device_t device;
    };

    // struct mapped_region {
    //     container_cb region;
    //     std::unique_ptr<interface::mapping_descriptor> md;
    // };


} // namespace sardine
