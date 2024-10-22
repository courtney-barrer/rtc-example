#pragma once

#include <baldr/dm.hpp>

namespace baldr::fakedm
{
    std::unique_ptr<interface::DM> make_dm(json::object config);

} // namespace baldr::fakedm
