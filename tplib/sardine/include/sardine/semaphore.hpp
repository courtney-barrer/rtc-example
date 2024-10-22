#pragma once

#include <boost/interprocess/sync/interprocess_semaphore.hpp>

namespace sardine
{

    using semaphore_t = myboost::interprocess::interprocess_semaphore;
    using mutex_t = myboost::interprocess::interprocess_mutex;

} // namespace sardine
