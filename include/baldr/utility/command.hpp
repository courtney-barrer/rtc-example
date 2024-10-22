#pragma once

#include <fmt/core.h>

#include <future>
#include <atomic>

namespace baldr
{

    enum class cmd
    {
        pause,
        run,
        exit
    };

    using Command = std::atomic<cmd>;

    template<typename F>
    std::future<void> spawn_runner(F component, Command& command, std::string component_name) {
        return std::async(std::launch::async, [comp = std::move(component), &command, component_name] () mutable {
            command.store(cmd::pause);

            fmt::print("starting loop for {}\n", component_name);

            while (true) {
                auto new_cmd = command.load();
                if (new_cmd == cmd::exit)
                    break;

                if (new_cmd == cmd::pause) {
                    // fmt::print("pausing loop for {}\n", component_name);
                    // command.wait(cmd::pause);
                    // fmt::print("exiting pause for {}\n", component_name);
                    continue;
                }

                comp();
            }

            fmt::print("exiting loop for {}\n", component_name);

        });
    }

} // namespace baldr
