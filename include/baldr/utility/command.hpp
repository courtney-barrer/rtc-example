#pragma once

#include <baldr/type.hpp>

#include <emu/assert.hpp>
#include <emu/cstring_view.hpp>

#include <fmt/core.h>

#include <future>
#include <atomic>
#include <mutex>

namespace baldr
{

    enum class Command
    {
        pause,
        run,
        step,
        exit
    };

    inline emu::cstring_view format_as(Command c) {
        switch (c) {
            case Command::pause: return "pause";
            case Command::run:   return "run";
            case Command::step:  return "step";
            case Command::exit:  return "exit";
        }
        EMU_UNREACHABLE;
    }

    enum class Status
    {
        none,
        acquired,
        running,
        pausing,
        exited,
        crashed
    };

    inline emu::cstring_view format_as(Status s) {
        switch (s) {
            case Status::none:     return "none";
            case Status::acquired: return "acquired";
            case Status::running:  return "running";
            case Status::pausing:  return "pausing";
            case Status::exited:   return "exited";
            case Status::crashed:  return "crashed";
        }
        EMU_UNREACHABLE;
    }

    // enum class cmd
    // {
    //     pause,
    //     run,
    //     step,
    //     exit
    // };

    // inline emu::cstring_view format_as(cmd c) {
    //     switch (c) {
    //         case cmd::pause: return "pause";
    //         case cmd::run:   return "run";
    //         case cmd::step:  return "step";
    //         case cmd::exit:  return "exit";
    //     }
    //     EMU_UNREACHABLE;
    // }

} // namespace baldr
