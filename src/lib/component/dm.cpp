#include <baldr/component/dm.hpp>

#include <baldr/component/dm/bmc.hpp>
#include <baldr/component/dm/fakedm.hpp>

#include <baldr/utility/runner.hpp>

#include <stdexcept>

namespace baldr
{

    std::unique_ptr<interface::DM> make_dm(string type, json::object config) {
        if (type == "bmc") return bmc::make_dm(config);

        if (type == "fake") return fakedm::make_dm(config);

        throw std::runtime_error("Could not instantiate the DM");
    }

namespace node
{

    DM::DM(string type, json::object config, commands_consumer_t commands, SpinLock& lock, size_t idx)
        : dm_impl(make_dm(type, config))
        , commands(std::move(commands))
        , lock(&lock)
        , idx(idx)
    {}

    void DM::operator()() {
        // waiting on the commands lock
        lock->lock(idx);

        // ingore for now
        commands.recv(ctx);

        // read the commands shm and send them to the DM
        dm_impl->send_command(commands.view());
    }

} // namespace node

     node::DM init_dm(string type, json::object config) {

        auto dm_config = config.at("config").as_object();

        // TODO: adds logs for these two.
        auto commands_url = sardine::json::opt_to<url>(config.at("io"), "commands").value();
        auto mutex_url = sardine::json::opt_to<url>(config.at("sync"), "wait").value();

        commands_consumer_t commands = EMU_UNWRAP_RES_OR_THROW_LOG(commands_consumer_t::open(commands_url),
            "Could not open commands using url: {}", commands_url);

        SpinLock& lock = EMU_UNWRAP_RES_OR_THROW_LOG(sardine::from_url<SpinLock>(mutex_url),
          "Could not open wait lock using url: {}", mutex_url);

        auto idx = json::opt_to<size_t>(config.at("sync"), "idx").value_or(0);

        return node::DM(type, dm_config, std::move(commands), lock, idx);
    }

    std::future<void> init_dm_thread(ComponentInfo& ci, string type, json::object config) {
        auto dm = init_dm(type, config);

        return spawn_runner(std::move(dm), ci);
    }

} // namespace baldr
