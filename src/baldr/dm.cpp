#include <baldr/dm.hpp>
#include <stdexcept>

#include <baldr/component/dm/bmc.hpp>
#include <baldr/component/dm/fakedm.hpp>

namespace baldr
{

    std::unique_ptr<interface::DM> make_dm(string type, json::object config) {
        if (type == "bmd") return bmc::make_dm(config);

        if (type == "fake") return fakedm::make_dm(config);

        throw std::runtime_error("Could not instantiate the DM");
    }

namespace node
{

    DM::DM(string type, json::object config, commands_consumer_t commands, sardine::mutex_t& mutex)
        : dm_impl(make_dm(type, config))
        , commands(std::move(commands))
        , mutex(&mutex)
    {}

    void DM::operator()() {
        mutex->lock();

        commands.recv(ctx);

        dm_impl->send_command(commands.view());
    }

} // namespace node

     std::future<void> init_dm(json::object config) {
        auto type = sardine::json::opt_to<std::string>(config, "type").value();

        auto dm_config = config.at("config").as_object();

        // TODO: adds logs for these two.
        auto commands_url = sardine::json::opt_to<url>(config.at("io"), "commands").value();
        auto mutex_url = sardine::json::opt_to<url>(config.at("sync"), "wait").value();

        commands_consumer_t commands = EMU_UNWRAP_OR_THROW_LOG(commands_consumer_t::open(commands_url),
            "Could not open commands using url: {}", commands_url);

        sardine::mutex_t& mutex = EMU_UNWRAP_OR_THROW_LOG(sardine::from_url<sardine::mutex_t>(mutex_url),
          "Could not open wait mutex using url: {}", mutex_url);

        node::DM dm(type, dm_config, std::move(commands), mutex);

        Command& command = sardine::from_url<Command>(*sardine::json::opt_to<url>(config, "command")).value();

        return spawn_runner(std::move(dm), command, fmt::format("dm {}", type));
    }

} // namespace baldr
