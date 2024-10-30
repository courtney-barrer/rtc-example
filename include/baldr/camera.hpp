#pragma once

#include "baldr/utility/spin_lock.hpp"
#include <baldr/type.hpp>

#include <cstdint>
#include <sardine/sardine.hpp>
#include <sardine/semaphore.hpp>

#include <span>
#include <memory>

namespace baldr
{

    struct CameraLogic
    {

        // std::shared_ptr<interface::Camera> camera_impl;

        frame_producer_t frame;
        SpinLock* lock;
        sardine::host_context ctx = {};
    };

namespace interface
{

    struct Camera
    {

        CameraLogic cam_logic;

        Camera(CameraLogic cam_logic);

        virtual ~Camera() = default;

        virtual void set_command(cmd new_command) = 0;
        virtual std::span<const uint16_t> last_frame() const = 0;

        void send_frame(std::span<const uint16_t> last_frame) {
            std::ranges::copy(last_frame, cam_logic.frame.view().data());

            cam_logic.frame.send(cam_logic.ctx);
            cam_logic.lock->unlock();
        };

    };

} // namespace interface

    std::unique_ptr<interface::Camera> make_camera(string type, CameraLogic cam_logic, json::object config);

namespace node
{

    struct Camera
    {

        std::shared_ptr<interface::Camera> camera_impl;

        Camera(string type, CameraLogic cam_logic, json::object config);

        void set_command(cmd new_command);

        std::span<const uint16_t> last_frame() const;

    };

} // namespace node

    node::Camera init_camera(json::object config);
    std::future<void> init_camera_thread(json::object config);

} // namespace baldr
