#pragma once

#include <baldr/camera.hpp>

namespace baldr::fakecam
{

    std::unique_ptr<interface::Camera> make_camera(json::object config);

} // namespace baldr::fakecam
