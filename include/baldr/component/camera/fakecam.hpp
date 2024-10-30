#pragma once

#include <baldr/camera.hpp>

namespace baldr::fakecam
{

    std::unique_ptr<interface::Camera> make_camera(CameraLogic cam_logic, json::object config);

} // namespace baldr::fakecam
