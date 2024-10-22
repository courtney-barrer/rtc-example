#pragma once

#include <baldr/dm.hpp>

namespace baldr::bmc
{
    std::unique_ptr<interface::DM> make_dm(json::object config);

} // namespace baldr::bmc
