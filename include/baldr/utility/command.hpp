#pragma once

#include <emu/assert.hpp>

#include <fmt/core.h>

#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <boost/atomic/ipc_atomic.hpp>


#include <future>
#include <atomic>
#include <mutex>

namespace baldr
{

    enum class cmd
    {
        pause,
        run,
        step,
        exit
    };

    inline std::string_view format_as(cmd c) {
        switch (c) {
            case cmd::pause: return "pause";
            case cmd::run: return "run";
            case cmd::step: return "step";
            case cmd::exit: return "exit";
        }
        EMU_UNREACHABLE;
    }



    // struct Sync {
    //     myboost::interprocess::interprocess_mutex      mutex;
    //     myboost::interprocess::interprocess_condition  cond;
    // };

    // struct Command : Sync {

    //     std::atomic<cmd> c;

    //     Command() = default;
    //     Command(cmd c)
    //         : c(c)
    //     {}

    //     void send(cmd new_c) {
    //         c.store(new_c);
    //         this->cond.notify_one();
    //     }

    //     cmd recv() const {
    //         return c.load();
    //     }

    //     void wait_unpause() {
    //         myboost::interprocess::scoped_lock<myboost::interprocess::interprocess_mutex> lock(this->mutex);

    //         this->cond.wait(lock, [&]{ return recv() != cmd::pause; });
    //     }

    //     bool operator==(cmd other) const {
    //         return recv() == other;
    //     }

    // };

    using Command = myboost::ipc_atomic<cmd>;

    template<typename F>
    std::future<void> spawn_runner(F component, Command& command, std::string component_name) {
        return std::async(std::launch::async, [comp = std::move(component), &command, component_name] () mutable {
            // command.store(cmd::pause);

            fmt::print("starting loop for {}\n", component_name);

            while (true) {
                auto new_cmd = command.load();

                if (new_cmd == cmd::pause) {
                    command.wait(cmd::pause);
                    continue;
                }

                if (new_cmd == cmd::exit)
                    break;

                comp();

                if (new_cmd == cmd::step)
                   command = cmd::pause;
            }

            fmt::print("exiting loop for {}\n", component_name);

        });
    }

} // namespace baldr
