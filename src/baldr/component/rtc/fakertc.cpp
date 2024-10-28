#include <baldr/component/rtc/fakertc.hpp>
#include <baldr/utility/updatable.hpp>

#include <algorithm>

namespace baldr::fakertc
{

    struct RTC : interface::RTC
    {
        updatable<double> factor;
        updatable<double> offset;

        RTC(json::object config)
            : factor(json::object_to<double>     (config, "factor"))
            , offset(json::object_to<double>     (config, "offset"))
        {}

        void compute(std::span<const uint16_t> frame, std::span<double> signal) {

                std::ranges::transform(frame, signal.data(),
                    [
                        factor = factor.current(),
                        offset = offset.current()
                    ] (auto pixel ) -> double {
                        return static_cast<double>(pixel) * factor + offset;
                    });
        }

    };

    std::unique_ptr<interface::RTC> make_rtc(json::object config) {
        return std::make_unique<RTC>(config);
    }

} // namespace baldr::fakertc
