#pragma once

#include <sardine/config.hpp>

#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <sardine/utility/sync/barrier.hpp>

namespace sardine::sync
{

    using semaphore_t = boost::interprocess::interprocess_semaphore;
    using mutex_t = boost::interprocess::interprocess_mutex;

    using scoped_lock = boost::interprocess::scoped_lock<mutex_t>;
} // namespace sardine::sync
