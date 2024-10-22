#include <baldr/rtc.hpp>

#include <baldr/component/rtc/ben.hpp>

#include <stdexcept>

namespace baldr
{
    std::unique_ptr<interface::RTC> make_rtc(string type, json::object config) {
        if (type == "ben") return benrtc::make_rtc(config);

        throw std::runtime_error("Could not instantiate the RTC");
    }

namespace node
{

    RTC::RTC(string type, json::object config, frame_consumer_t frame, commands_producer_t commands, sardine::mutex_t& wait_mutex, sardine::mutex_t& notify_mutex)
        : rtc_impl(make_rtc(type, config))
        , frame(std::move(frame))
        , commands(std::move(commands))
        , wait_mutex(&wait_mutex)
        , notify_mutex(&notify_mutex)
    {}

    void RTC::operator()() {
        wait_mutex->lock();

        frame.recv(ctx);

        rtc_impl->compute(frame.view(), commands.view());

        commands.send(ctx);

        notify_mutex->unlock();
    }

} // namespace node

    std::future<void> init_rtc(json::object config) {
        auto type = sardine::json::opt_to<std::string>(config, "type").value();

        auto camera_config = config.at("config").as_object();

        // TODO: adds logs for these two.
        auto frame_url = sardine::json::opt_to<url>(config.at("io"), "frame").value();
        auto commands_url = sardine::json::opt_to<url>(config.at("io"), "commands").value();

        auto wait_mutex_url = sardine::json::opt_to<url>(config.at("sync"), "wait").value();
        auto notify_mutex_url = sardine::json::opt_to<url>(config.at("sync"), "notify").value();

        frame_consumer_t frame = EMU_UNWRAP_OR_THROW_LOG(frame_consumer_t::open(frame_url),
            "Could not open frame using url: {}", frame_url);

        commands_producer_t commands = EMU_UNWRAP_OR_THROW_LOG(commands_producer_t::open(commands_url),
            "Could not open commands using url: {}", commands_url);

        sardine::mutex_t& wait_mutex = EMU_UNWRAP_OR_THROW_LOG(sardine::from_url<sardine::mutex_t>(wait_mutex_url),
          "Could not open wait mutex using url: {}", wait_mutex_url);

        sardine::mutex_t& notify_mutex = EMU_UNWRAP_OR_THROW_LOG(sardine::from_url<sardine::mutex_t>(notify_mutex_url),
          "Could not open notify mutex using url: {}", notify_mutex_url);

        node::RTC rtc(type, camera_config, std::move(frame), std::move(commands), wait_mutex, notify_mutex);

        Command& command = sardine::from_url<Command>(*sardine::json::opt_to<url>(config, "command")).value();

        return spawn_runner(std::move(rtc), command, fmt::format("rtc {}", type));
    }

} // namespace baldr
