#pragma once

#include "baldr/utility/command.hpp"
#include <baldr/utility/component_info.hpp>

#include <emu/macro.hpp>

#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/atomic/ipc_atomic.hpp>

#include <thread>
#include <atomic>

namespace baldr
{
    // using CommandAtom = boost::ipc_atomic<cmd>;

    template<typename F>
    std::future<void> spawn_runner(F component, ComponentInfo& ci) {
        return std::async(std::launch::async, [comp = std::move(component), &ci] () mutable {
            // command.store(cmd::pause);

            fmt::print("starting loop for {}\n", ci.name());

            ci.set_status(Status::running);

            while (true) {

                auto new_cmd = ci.cmd();

                if (new_cmd == Command::pause) {
                    ci.set_status(Status::pausing);
                    ci.wait_unpause();
                    ci.set_status(Status::running);
                    continue;
                }

                if (new_cmd == Command::exit)
                    break;

                comp();
                ci.loop_count.add(1);

                if (new_cmd == Command::step)
                   ci.set_cmd(Command::pause);
            }

            ci.set_status(Status::exited);

            fmt::print("exiting loop for {}\n", ci.name());

        });
    }


    // template<typename T>
    // void run_async(std::stop_token stop_token, std::atomic<bool>& command, T& obj)
    // {
    //     std::size_t count = 0;
    //     while (!stop_token.stop_requested()) {

    //         if (command) {
    //             obj.compute();
    //         } else {
    //             command.wait(false);
    //         }
    //     }
    // }

    // struct async_runner
    // {
    //     std::atomic<bool> command;
    //     std::jthread t;

    //     void pause() {
    //         command = false;
    //     }

    //     /// Resumes the execution of RTC operations.
    //     void resume() {
    //         command = true;
    //         command.notify_one();
    //     }

    //     /// Starts the execution of RTC operations asynchronously.
    //     template<typename T>
    //     void start(T& component) {
    //         if (t.joinable())
    //             stop();

    //         command = true;
    //         t = std::jthread(run_async<T>, std::ref(command), std::ref(component));
    //     }

    //     /// Stops the execution of RTC operations.
    //     void stop() {
    //         t.request_stop();
    //         resume(); // in case thread is waiting on command.
    //         t.join();
    //     }


    // };



} // namespace baldr
