#pragma once

#include <baldr/rtc.hpp>

namespace baldr::fakertc
{
    std::unique_ptr<interface::RTC> make_rtc(json::object config);

} // namespace baldr::fakertc
