#pragma once

#include <baldr/type.hpp>
#include <baldr/utility/component_info.hpp>

#include <sardine/sardine.hpp>
#include <sardine/utility/sync/sync.hpp>

#include <memory>
#include <span>

namespace baldr
{

namespace interface
{

    struct RTC
    {

        virtual ~RTC() = default;

        virtual void compute(std::span<const uint16_t> frame, std::span<double> commands) = 0;

    };

} // namespace interface

    std::unique_ptr<interface::RTC> make_rtc(string type, json::object config);

namespace node
{

    struct RTC
    {

        std::unique_ptr<interface::RTC> rtc_impl;

        frame_consumer_t frame;
        commands_producer_t commands;
        SpinLock* wait_lock;
        size_t wait_idx;
        SpinLock* notify_lock;

        sardine::host_context ctx;

        RTC(
            string type, json::object config,
            frame_consumer_t frame, commands_producer_t commands,
            SpinLock& wait_lock, size_t wait_idx, SpinLock& notify_lock
        );

        void operator()();

    };

} // namespace node

    node::RTC init_rtc(string type, json::object config);
    std::future<void> init_rtc_thread(ComponentInfo& ci, string type, json::object config);


} // namespace baldr
