#pragma once

#include <baldr/component/camera.hpp>

namespace baldr::fakecam
{

    std::future<void> make_camera(ComponentInfo& ci, CameraLogic cam_logic, json::object config, bool async);

} // namespace baldr::fakecam
