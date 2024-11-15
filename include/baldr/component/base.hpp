#pragma once

#include <baldr/config.hpp>
#include <baldr/type.hpp>

#include <memory>
#include <future>

namespace baldr
{

    std::future<void> init_component_thread(json::object config);

} // namespace baldr
