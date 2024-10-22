#pragma once

#include <baldr/type.hpp>

#include <sardine/sardine.hpp>
#include <sardine/semaphore.hpp>

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
        sardine::mutex_t* wait_mutex;
        sardine::mutex_t* notify_mutex;

        sardine::host_context ctx;

        RTC(string type, json::object config, frame_consumer_t frame, commands_producer_t commands, sardine::mutex_t& wait_mutex, sardine::mutex_t& notify_mutex);

        void operator()();

    };

} // namespace node

    std::future<void> init_rtc(json::object config);

} // namespace baldr
