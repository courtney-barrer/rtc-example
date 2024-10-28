#include <baldr/rtc.hpp>

#include <baldr/component/rtc/ben.hpp>
#include <baldr/component/rtc/fakertc.hpp>

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

    RTC::RTC(string type, json::object config, frame_consumer_t frame, commands_producer_t commands, SpinLock& wait_lock, SpinLock& notify_lock)
        : rtc_impl(make_rtc(type, config))
        , frame(std::move(frame))
        , commands(std::move(commands))
        , wait_lock(&wait_lock)
        , notify_lock(&notify_lock)
    {}

    void RTC::operator()() {
        wait_lock->lock();

        frame.recv(ctx);

        rtc_impl->compute(frame.view(), commands.view());

        commands.send(ctx);

        notify_lock->unlock();
    }

} // namespace node

    node::RTC init_rtc(json::object config) {

        auto type = sardine::json::opt_to<std::string>(config, "type").value();

        auto camera_config = config.at("config").as_object();

        // TODO: adds logs for these two.
        auto frame_url = sardine::json::opt_to<url>(config.at("io"), "frame").value();
        auto commands_url = sardine::json::opt_to<url>(config.at("io"), "commands").value();

        auto wait_lock_url = sardine::json::opt_to<url>(config.at("sync"), "wait").value();
        auto notify_lock_url = sardine::json::opt_to<url>(config.at("sync"), "notify").value();

        frame_consumer_t frame = EMU_UNWRAP_OR_THROW_LOG(frame_consumer_t::open(frame_url),
            "Could not open frame using url: {}", frame_url);

        commands_producer_t commands = EMU_UNWRAP_OR_THROW_LOG(commands_producer_t::open(commands_url),
            "Could not open commands using url: {}", commands_url);

        SpinLock& wait_lock = EMU_UNWRAP_OR_THROW_LOG(sardine::from_url<SpinLock>(wait_lock_url),
          "Could not open wait lock using url: {}", wait_lock_url);

        SpinLock& notify_lock = EMU_UNWRAP_OR_THROW_LOG(sardine::from_url<SpinLock>(notify_lock_url),
          "Could not open notify lock using url: {}", notify_lock_url);

        return node::RTC(type, camera_config, std::move(frame), std::move(commands), wait_lock, notify_lock);
    }

    std::future<void> init_rtc_thread(json::object config) {
        auto rtc = init_rtc(config);

        Command& command = sardine::from_url<Command>(*sardine::json::opt_to<url>(config, "command")).value();

        return spawn_runner(std::move(rtc), command, "rtc");
    }

} // namespace baldr
