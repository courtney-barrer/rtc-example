#pragma once

#include <baldr/component/dm.hpp>

namespace baldr::fakedm
{
    std::unique_ptr<interface::DM> make_dm(json::object config);

} // namespace baldr::fakedm
