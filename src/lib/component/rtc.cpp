#include <baldr/component/rtc.hpp>
#include <baldr/component/rtc/ben.hpp>
#include <baldr/component/rtc/fakertc.hpp>
#include <baldr/utility/runner.hpp>

#include <iostream>
#include <stdexcept>

namespace baldr
{
    std::unique_ptr<interface::RTC> make_rtc(string type, json::object config) {
        if (type == "ben") return benrtc::make_rtc(config);

        if (type == "fake") return fakertc::make_rtc(config);

        throw std::runtime_error("Could not instantiate the RTC");
    }

namespace node
{

    RTC::RTC(string type, json::object config, frame_consumer_t frame, commands_producer_t commands,
             SpinLock& wait_lock, size_t wait_idx, SpinLock& notify_lock)
        : rtc_impl(make_rtc(type, config))
        , frame(std::move(frame))
        , commands(std::move(commands))
        , wait_lock(&wait_lock)
        , wait_idx(wait_idx)
        , notify_lock(&notify_lock)
    {}

    void RTC::operator()() {
        // waiting on the frame lock
        wait_lock->lock(wait_idx);
        // ingore for now
        frame.recv(ctx);

        // read the frame shm, compute the commands and write them
        // to the commands shm
        rtc_impl->compute(frame.view(), commands.view());

        // ignore for now
        commands.send(ctx);

        // unlock the commands lock
        notify_lock->unlock();
    }

} // namespace node

    node::RTC init_rtc(string type, json::object config) {

        auto camera_config = config.at("config").as_object();

        // TODO: adds logs for these two.
        auto frame_url = sardine::json::opt_to<url>(config.at("io"), "frame").value();
        auto commands_url = sardine::json::opt_to<url>(config.at("io"), "commands").value();

        auto wait_lock_url = sardine::json::opt_to<url>(config.at("sync"), "wait").value();
        auto notify_lock_url = sardine::json::opt_to<url>(config.at("sync"), "notify").value();

        frame_consumer_t frame = EMU_UNWRAP_RES_OR_THROW_LOG(frame_consumer_t::open(frame_url),
            "Could not open frame using url: {}", frame_url);

        commands_producer_t commands = EMU_UNWRAP_RES_OR_THROW_LOG(commands_producer_t::open(commands_url),
            "Could not open commands using url: {}", commands_url);

        SpinLock& wait_lock = EMU_UNWRAP_RES_OR_THROW_LOG(sardine::from_url<SpinLock>(wait_lock_url),
          "Could not open wait lock using url: {}", wait_lock_url);

        auto wait_idx = json::opt_to<size_t>(config.at("sync"), "wait_idx").value_or(0);

        SpinLock& notify_lock = EMU_UNWRAP_RES_OR_THROW_LOG(sardine::from_url<SpinLock>(notify_lock_url),
          "Could not open notify lock using url: {}", notify_lock_url);

        return node::RTC(type, camera_config, std::move(frame), std::move(commands), wait_lock, wait_idx, notify_lock);
    }

    std::future<void> init_rtc_thread(ComponentInfo& ci, string type, json::object config) {
        auto rtc = init_rtc(type, config);

        return spawn_runner(std::move(rtc), ci);
    }

} // namespace baldr
