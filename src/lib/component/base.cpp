#include <baldr/component/base.hpp>
#include <baldr/component/camera.hpp>
#include <baldr/component/rtc.hpp>
#include <baldr/component/dm.hpp>

#include <baldr/utility/command.hpp>
#include <baldr/utility/component_info.hpp>
#include <sardine/url.hpp>
#include <sardine/json.hpp>

namespace baldr
{

    std::future<void> init_component_thread(json::object config) {
        // auto commands_url = sardine::json::opt_to<url>(config, "cinfo").value();

        ComponentInfo& component_info = sardine::from_json<ComponentInfo&>(config.at("cinfo")).value();

        component_info.acquire();

        auto complete_type = sardine::json::opt_to<std::string>(config, "type").value();

        // Get the first part of the type until ":"
        auto pos = complete_type.find(":");
        if (pos != std::string::npos) {
            auto type_head = complete_type.substr(0, pos);
            auto type_tail = complete_type.substr(pos + 1);

            if (type_head == "camera") return init_camera_thread(component_info, type_tail, config);

            if (type_head == "rtc") return init_rtc_thread(component_info, type_tail, config);

            if (type_head == "dm") return init_dm_thread(component_info, type_tail, config);
        }

        throw std::runtime_error(fmt::format("Unknown component type: {}", complete_type));

        // auto component = make_component(complete_type, config);

        // Command& command = sardine::from_url<Command>(*sardine::json::opt_to<url>(config, "command")).value();

        // return spawn_runner([c = std::move(component)]{
        //     c->compute();
        // }, command, complete_type);
    }


} // namespace baldr
