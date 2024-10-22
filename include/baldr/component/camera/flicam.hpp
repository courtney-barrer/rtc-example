#pragma once

#include <baldr/camera.hpp>

namespace baldr::flicam
{

    std::unique_ptr<interface::Camera> make_camera(json::object config);

} // namespace baldr::flicam
