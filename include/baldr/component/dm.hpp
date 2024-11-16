#pragma once

#include <baldr/type.hpp>
#include <baldr/utility/component_info.hpp>

#include <sardine/sardine.hpp>
#include <sardine/utility/sync/sync.hpp>

#include <memory>
#include <span>

namespace baldr
{

namespace interface
{

    struct DM
    {

        virtual ~DM() = default;

        virtual void send_command(span<const double> commands) = 0;

    };

} // namespace interface

    std::unique_ptr<interface::DM> make_dm(json::object config);

namespace node
{

    struct DM
    {

        std::unique_ptr<interface::DM> dm_impl;

        commands_consumer_t commands;
        SpinLock* lock;
        size_t idx;
        sardine::host_context ctx;

        DM(string type, json::object config, commands_consumer_t commands, SpinLock& lock, size_t idx);

        void operator()();

    };

} // namespace node

    node::DM init_dm(string type, json::object config);
    std::future<void> init_dm_thread(ComponentInfo& ci, string type, json::object config);

} // namespace baldr