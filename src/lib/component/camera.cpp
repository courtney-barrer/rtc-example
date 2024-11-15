#include <baldr/type.hpp>
#include <baldr/component/camera.hpp>
#include <sardine/context.hpp>
#include <stdexcept>

#include <baldr/utility/command.hpp>
#include <baldr/component/camera/flicam.hpp>
#include <baldr/component/camera/fakecam.hpp>

#include <baldr/utility/runner.hpp>

#include <fmt/core.h>

namespace baldr
{
    std::future<void> make_camera(string type, ComponentInfo& ci, CameraLogic cam_logic, json::object config, bool async) {
        if(type == "fli") return flicam::make_camera(ci, std::move(cam_logic), config, async);

        if(type == "fake") return fakecam::make_camera(ci, std::move(cam_logic), config, async);

        throw std::runtime_error(fmt::format("Could not instantiate the camera {}", type));
    }

namespace interface
{

        Camera::Camera(CameraLogic cam_logic)
            : cam_logic(std::move(cam_logic))
        {}

} // namespace interface

// namespace node
// {

//     Camera::Camera(string type, CameraLogic cam_logic, json::object config)
//         : camera_impl(make_camera(type, std::move(cam_logic), config))
//     {}

//     void Camera::set_command(Command new_command) {
//         camera_impl->set_command(new_command);
//     }

//     std::span<const uint16_t> Camera::get_last_frame() const {
//         return camera_impl->get_last_frame();
//     }


//     // void Camera::operator()() {

//     //     camera_impl->get_frame(frame.view());

//     //     frame.send(ctx);
//     //     lock->unlock();
//     // }

// } // namespace node

    // node::Camera init_camera(string type, json::object config) {
    //     auto camera_config = config.at("config").as_object();

    //     // TODO: adds logs for these two.
    //     auto frame_url = sardine::json::opt_to<url>(config.at("io"), "frame").value();
    //     auto lock_url = sardine::json::opt_to<url>(config.at("sync"), "notify").value();

    //     frame_producer_t frame = EMU_UNWRAP_RES_OR_THROW_LOG(frame_producer_t::open(frame_url),
    //         "Could not open frame using url: {}", frame_url);

    //     SpinLock& lock = EMU_UNWRAP_RES_OR_THROW_LOG(sardine::from_url<SpinLock>(lock_url),
    //         "Could not open notify lock using url: {}", lock_url);

    //     bool async = sardine::json::opt_to<bool>(config, "async").value_or(false);

    //     CameraLogic cam_logic{
    //         .frame = std::move(frame),
    //         .lock = &lock,
    //         async
    //     };

    //     return node::Camera(type, std::move(cam_logic), camera_config);
    //  }

     std::future<void> init_camera_thread(ComponentInfo& ci, string type, json::object config) {
        auto camera_config = config.at("config").as_object();

        // TODO: adds logs for these two.
        auto frame_url = sardine::json::opt_to<url>(config.at("io"), "frame").value();
        auto lock_url = sardine::json::opt_to<url>(config.at("sync"), "notify").value();

        frame_producer_t frame = EMU_UNWRAP_RES_OR_THROW_LOG(frame_producer_t::open(frame_url),
            "Could not open frame using url: {}", frame_url);

        SpinLock& lock = EMU_UNWRAP_RES_OR_THROW_LOG(sardine::from_url<SpinLock>(lock_url),
            "Could not open notify lock using url: {}", lock_url);

        bool async = sardine::json::opt_to<bool>(config, "async").value_or(false);

        CameraLogic cam_logic{
            .frame = std::move(frame),
            .lock = &lock
        };
        // auto camera =  node::Camera(type, std::move(cam_logic), camera_config);

        return make_camera(type, ci, std::move(cam_logic), camera_config, async);


        // // return spawn_runner(std::move(camera), command, "camera");
        // return std::async(std::launch::async, [camera = std::move(camera), &ci] () mutable {
        //     fmt::print("starting loop for {}\n", ci.name());

        //     ci.set_status(Status::pausing);

        //     auto new_cmd = Command::pause;
        //     while (true) {
        //         new_cmd = ci.wait_new_cmd(new_cmd);

        //         auto new_cmd = ci.cmd();
        //         camera.set_command(new_cmd);

        //         // If it was exit, we just exit the while loop and end the thread.
        //         if (new_cmd == Command::exit)
        //             break;

        //         // If it was step, we considere it done and we are now back to pause mode
        //         if (new_cmd == Command::step) {
        //             ci.set_cmd(Command::pause);
        //             continue;
        //         }
        //     }

        //     ci.set_status(Status::exited);

        //     fmt::print("exiting loop for {}\n", ci.name());

        // });
    }

} // namespace baldr
