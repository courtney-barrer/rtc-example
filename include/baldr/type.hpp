#pragma once

#include <baldr/config.hpp>
#include <baldr/utility/command.hpp>
#include <baldr/utility/spin_lock.hpp>

#include <sardine/fwd.hpp>
#include <sardine/type/json.hpp>
#include <sardine/type/url.hpp>
#include <sardine/buffer.hpp>

#include <emu/error.hpp>
#include <emu/cstring_view.hpp>

#include <span>
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace baldr
{

    using std::size_t, std::span, std::string, std::string_view, std::vector;

    using emu::result;
    using emu::cstring_view;

    using sardine::url_view, sardine::url;

    namespace json = sardine::json;
    namespace urls = sardine::urls;

    using frame_producer_t = sardine::producer<std::span<uint16_t>>;
    using frame_consumer_t = sardine::consumer<std::span<uint16_t>>;

    using commands_producer_t = sardine::producer<std::span<double>>;
    using commands_consumer_t = sardine::consumer<std::span<double>>;

} // namespace baldr
