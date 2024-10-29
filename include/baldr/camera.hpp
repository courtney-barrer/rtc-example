#pragma once

#include "baldr/utility/spin_lock.hpp"
#include <baldr/type.hpp>

#include <sardine/sardine.hpp>
#include <sardine/semaphore.hpp>

#include <span>
#include <memory>

namespace baldr
{

namespace interface
{

    struct Camera
    {

        virtual ~Camera() = default;

        virtual bool get_frame(std::span<uint16_t> frame) = 0;
    };

} // namespace interface

    std::unique_ptr<interface::Camera> make_camera(string type, json::object config);

namespace node
{

    struct Camera
    {

        std::shared_ptr<interface::Camera> camera_impl;

        frame_producer_t frame;
        SpinLock* lock;
        sardine::host_context ctx;

        Camera(string type, json::object config, frame_producer_t frame, SpinLock& lock);

        void operator()();

    };

} // namespace node

    node::Camera init_camera(json::object config);
    std::future<void> init_camera_thread(json::object config);

} // namespace baldr
