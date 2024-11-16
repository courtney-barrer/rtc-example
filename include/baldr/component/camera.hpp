#pragma once

#include <baldr/utility/spin_lock.hpp>
#include <baldr/type.hpp>
#include <baldr/utility/component_info.hpp>
#include <baldr/utility/runner.hpp>

#include <cstdint>
#include <range/v3/range/conversion.hpp>
#include <sardine/sardine.hpp>
#include <sardine/utility/sync/sync.hpp>

#include <span>
#include <memory>

namespace baldr
{

    struct CameraLogic
    {

        // std::shared_ptr<interface::Camera> camera_impl;

        frame_producer_t frame;
        SpinLock* lock;
        bool async;
        sardine::host_context ctx = {};
    };

namespace interface
{

    struct Camera
    {

        CameraLogic cam_logic;

        Camera(CameraLogic cam_logic);
        Camera(Camera&&) = default;

        virtual ~Camera() = default;

        virtual void set_command(Command new_command) = 0;
        virtual std::span<const uint16_t> get_last_frame() const = 0;

        void send_frame(std::span<const uint16_t> last_frame) {
            std::ranges::copy(last_frame, cam_logic.frame.view().data());

            cam_logic.frame.send(cam_logic.ctx);
            cam_logic.lock->unlock();
        };

    };

} // namespace interface

    template<typename Camera>
    std::future<void> spawn_async_camera_runner(std::unique_ptr<Camera> component, ComponentInfo& ci) {
        return std::async(std::launch::async, [comp = std::move(component), &ci] () mutable {
            fmt::print("starting loop for {}\n", ci.name());

            ci.set_status(Status::running);

            auto new_cmd = Command::pause;
            while (true) {
                new_cmd = ci.wait_new_cmd(new_cmd);

                auto new_cmd = ci.cmd();
                comp->set_command(new_cmd);

                // If it was exit, we just exit the while loop and end the thread.
                if (new_cmd == Command::exit)
                    break;

                // If it was step, we considere it done and we are now back to pause mode
                if (new_cmd == Command::step) {
                    ci.set_cmd(Command::pause);
                    continue;
                }
            }

            ci.set_status(Status::exited);

            fmt::print("exiting loop for {}\n", ci.name());

        });
    }

    std::future<void> make_camera(ComponentInfo& ci, string type, CameraLogic cam_logic, json::object config);

// namespace node
// {

//     struct Camera
//     {

//         std::shared_ptr<interface::Camera> camera_impl;

//         Camera(string type, CameraLogic cam_logic, json::object config);

//         void set_command(Command new_command);

//         std::span<const uint16_t> get_last_frame() const;

//     };

// } // namespace node

    // node::Camera init_camera(string type, json::object config);
    std::future<void> init_camera_thread(ComponentInfo& ci, string type, json::object config);

} // namespace baldr
