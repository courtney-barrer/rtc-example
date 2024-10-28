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

    Camera::Camera(string type, json::object config, frame_producer_t frame, SpinLock& lock)
        : camera_impl(make_camera(type, config))
        , frame(std::move(frame))
        , lock(&lock)
    {
        fmt::print("yessay!\n");
    }

    void Camera::operator()() {

        camera_impl->get_frame(frame.view());

        frame.send(ctx);
        lock->unlock();
    }

} // namespace node

    node::Camera init_camera(json::object config) {
        auto type = sardine::json::opt_to<std::string>(config, "type").value();

        auto camera_config = config.at("config").as_object();

        // TODO: adds logs for these two.
        auto frame_url = sardine::json::opt_to<url>(config.at("io"), "frame").value();
        auto lock_url = sardine::json::opt_to<url>(config.at("sync"), "notify").value();

        frame_producer_t frame = EMU_UNWRAP_OR_THROW_LOG(frame_producer_t::open(frame_url),
            "Could not open frame using url: {}", frame_url);

        SpinLock& lock = EMU_UNWRAP_OR_THROW_LOG(sardine::from_url<SpinLock>(lock_url),
            "Could not open notify lock using url: {}", lock_url);

        return node::Camera(type, camera_config, std::move(frame), lock);
     }

     std::future<void> init_camera_thread(json::object config) {
        auto camera = init_camera(config);

        Command& command = sardine::from_url<Command>(*sardine::json::opt_to<url>(config, "command")).value();

        return spawn_runner(std::move(camera), command, "camera");
    }

} // namespace baldr
