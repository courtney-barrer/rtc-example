#pragma once

#include <emu/macro.hpp>

#include <thread>
#include <atomic>

namespace baldr
{

    template<typename T>
    void run_async(std::stop_token stop_token, std::atomic<bool>& command, T& obj)
    {
        std::size_t count = 0;
        while (!stop_token.stop_requested()) {

            if (command) {
                obj.compute();
            } else {
                command.wait(false);
            }
        }
    }

    struct async_runner
    {
        std::atomic<bool> command;
        std::jthread t;

        void pause() {
            command = false;
        }

        /// Resumes the execution of RTC operations.
        void resume() {
            command = true;
            command.notify_one();
        }

        /// Starts the execution of RTC operations asynchronously.
        template<typename T>
        void start(T& component) {
            if (t.joinable())
                stop();

            command = true;
            t = std::jthread(run_async<T>, std::ref(command), std::ref(component));
        }

        /// Stops the execution of RTC operations.
        void stop() {
            t.request_stop();
            resume(); // in case thread is waiting on command.
            t.join();
        }


    };



} // namespace baldr
