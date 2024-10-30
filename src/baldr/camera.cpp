#include <baldr/type.hpp>
#include <baldr/camera.hpp>
#include <sardine/context.hpp>
#include <stdexcept>

#include <baldr/utility/command.hpp>
#include <baldr/component/camera/flicam.hpp>
#include <baldr/component/camera/fakecam.hpp>

namespace baldr
{
    std::unique_ptr<interface::Camera> make_camera(string type, CameraLogic cam_logic, json::object config) {
        if(type == "fli") return flicam::make_camera(std::move(cam_logic), config);

        if(type == "fake") return fakecam::make_camera(std::move(cam_logic), config);

        throw std::runtime_error("Could not instantiate the camera");
    }

namespace interface
{

        Camera::Camera(CameraLogic cam_logic)
            : cam_logic(std::move(cam_logic))
        {}

} // namespace interface

namespace node
{

    Camera::Camera(string type, CameraLogic cam_logic, json::object config)
        : camera_impl(make_camera(type, std::move(cam_logic), config))
    {}

    void Camera::set_command(cmd new_command) {
        camera_impl->set_command(new_command);
    }

    std::span<const uint16_t> Camera::LOOK_last_frame() const {
        return camera_impl->LOOK_last_frame();
    }


    // void Camera::operator()() {

    //     camera_impl->get_frame(frame.view());

    //     frame.send(ctx);
    //     lock->unlock();
    // }

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

        CameraLogic cam_logic{
            .frame = std::move(frame),
            .lock = &lock
        };

        return node::Camera(type, std::move(cam_logic), camera_config);
     }

     std::future<void> init_camera_thread(json::object config) {
        auto camera = init_camera(config);

        Command& command = sardine::from_url<Command>(*sardine::json::opt_to<url>(config, "command")).value();

        // return spawn_runner(std::move(camera), command, "camera");
        return std::async(std::launch::async, [camera = std::move(camera), &command] () mutable {
            // command.store(cmd::pause);

            fmt::print("starting loop for {}\n", "camera");

            while (true) {
                auto new_cmd = command.load();
                camera.set_command(new_cmd);

                // If it was exit, we just exit the while loop and end the thread.
                if (new_cmd == cmd::exit)
                    break;

                // If it was step, we considere it done and we are now back to pause mode
                if (new_cmd == cmd::step) {
                    command = cmd::pause;
                    continue;
                }

                // Otherwise: run or pause. We stay in this state until it changes.
                auto last_cmd = new_cmd;
                command.wait(last_cmd);
            }

            fmt::print("exiting loop for {}\n", "camera");

        });
    }

} // namespace baldr
