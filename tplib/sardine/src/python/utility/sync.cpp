#include <sardine/utility/sync/sync.hpp>
#include <sardine/utility/sync/spin_lock.hpp>
#include <sardine/utility/sync/barrier.hpp>
#include <sardine/cache.hpp>
#include <sardine/python/url_helper.hpp>
#include <sardine/python/managed_helper.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace sardine
{

    void register_shm_sync(py::module m)
    {
        using namespace pybind11::literals;

        using namespace sync;

        // #############
        // # Semaphore #
        // #############

        py::class_<semaphore_t> semaphore(m, "Semaphore");

        semaphore
            .def("release", [](semaphore_t& sem, std::size_t count){
                for (std::size_t i = 0; i < count; ++i)
                    sem.post();
            }, py::arg("count") = 1)
            .def("acquire", [](semaphore_t& sem, bool blocking) -> bool {
                if (blocking) {
                    sem.wait();
                    return true;
                }
                else
                    return sem.try_wait();
            }, py::arg("blocking") = 0) // should be false be that provoke a segfault. No idea why
        ;

        register_url(semaphore);
        register_managed(semaphore)
            .add_init([](create_proxy proxy, int init_value) -> semaphore_t& {
                return proxy.create<semaphore_t>(init_value);
            }, "init_value"_a = 0);

        // #########
        // # Mutex #
        // #########

        py::class_<mutex_t> mutex(m, "Mutex");

        mutex
            .def("release", [](mutex_t& mut){
                    mut.unlock();
            })
            .def("acquire", [](mutex_t& mut, bool blocking) -> bool {
                if (blocking) {
                    mut.lock();
                    return true;
                }
                else
                    return mut.try_lock();
            }, py::arg("blocking") = true)
        ;

        register_url(mutex);
        register_managed(mutex)
            .add_default();

        // ############
        // # SpinLock #
        // ############

        py::enum_<LockState>(m, "LockState")
            .value("Locked", LockState::Locked)
            .value("Unlocked", LockState::Unlocked)
            .export_values();

        py::class_<SpinLock> spin_lock(m, "SpinLock");

        spin_lock
            .def("lock", &SpinLock::lock)
            .def("try_lock", &SpinLock::try_lock)
            .def("unlock", &SpinLock::unlock)
            .def_property_readonly("value", [](const SpinLock& sl) -> LockState {
                return sl.state.load();
            })
        ;

        register_url(spin_lock);
        register_managed(spin_lock)
            .add_init([](create_proxy proxy, LockState init_value) -> SpinLock& {
                return proxy.create<SpinLock>(init_value);
            }, "init_value"_a = LockState::Locked);

        // ###########
        // # Barrier #
        // ###########

        py::class_<Barrier> barrier(m, "Barrier");

        barrier
            .def("get_waiter", &Barrier::get_waiter)
            .def("get_notifier", &Barrier::get_notifier)
            .def("notify_all", &Barrier::notify_all)
        ;

        register_url(barrier);
        register_managed(barrier)
            .add_default();


        py::class_<Waiter>(m, "Waiter")
            .def("wait", &Waiter::wait)
        ;

        py::class_<Notifier>(m, "Notifier")
            .def("notify", &Notifier::notify)
        ;

    }


} // namespace sardine
