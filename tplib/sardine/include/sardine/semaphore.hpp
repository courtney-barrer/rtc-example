#pragma once

#include <boost/interprocess/sync/interprocess_semaphore.hpp>

namespace sardine
{

    using semaphore_t = boost::interprocess::interprocess_semaphore;
    using mutex_t = boost::interprocess::interprocess_mutex;

} // namespace sardine
