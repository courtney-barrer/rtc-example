#include <baldr/type.hpp>
#include <baldr/camera.hpp>
#include <sardine/context.hpp>
#include <stdexcept>

#include <baldr/utility/command.hpp>
#include <baldr/component/camera/flicam.hpp>
#include <baldr/component/camera/fakecam.hpp>

namespace baldr
{
     std::unique_ptr<interface::Camera> make_camera(string type, json::object config) {
         if(type == "fli") return flicam::make_camera(config);

         if(type == "fake") return fakecam::make_camera(config);

        throw std::runtime_error("Could not instantiate the camera");
     }

namespace node
{

     Camera::Camera(string type, json::object config, frame_producer_t frame, sardine::mutex_t& mutex)
          : camera_impl(make_camera(type, config))
          , frame(std::move(frame))
          , mutex(&mutex)
     {
          fmt::print("yessay!\n");
     }

     void Camera::operator()() {
          camera_impl->get_frame(frame.view());

          frame.send(ctx);
          mutex->unlock();
     }

} // namespace node

     std::future<void> init_camera(json::object config) {
        auto type = sardine::json::opt_to<std::string>(config, "type").value();

        auto camera_config = config.at("config").as_object();

        // TODO: adds logs for these two.
        auto frame_url = sardine::json::opt_to<url>(config.at("io"), "frame").value();
        auto mutex_url = sardine::json::opt_to<url>(config.at("sync"), "notify").value();

        frame_producer_t frame = EMU_UNWRAP_OR_THROW_LOG(frame_producer_t::open(frame_url),
            "Could not open frame using url: {}", frame_url);

        sardine::mutex_t& mutex = EMU_UNWRAP_OR_THROW_LOG(sardine::from_url<sardine::mutex_t>(mutex_url),
          "Could not open notify mutex using url: {}", mutex_url);

        node::Camera camera(type, camera_config, std::move(frame), mutex);

        Command& command = sardine::from_url<Command>(*sardine::json::opt_to<url>(config, "command")).value();

        return spawn_runner(std::move(camera), command, fmt::format("camera {}", type));
    }

} // namespace baldr
