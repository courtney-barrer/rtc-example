#pragma once

#include <baldr/component/rtc.hpp>

namespace baldr::fakertc
{
    std::unique_ptr<interface::RTC> make_rtc(json::object config);

} // namespace baldr::fakertc
