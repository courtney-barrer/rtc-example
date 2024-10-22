#pragma once

#include <baldr/rtc.hpp>

namespace baldr::benrtc
{
    std::unique_ptr<interface::RTC> make_rtc(json::object config);

} // namespace baldr::benrtc
