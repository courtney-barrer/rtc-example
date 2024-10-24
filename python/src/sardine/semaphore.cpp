#include <sardine/semaphore.hpp>
#include <sardine/cache.hpp>
#include <sardine/python/url_helper.hpp>

#include <pybind11/pybind11.h>

#include <semaphore>

namespace py = pybind11;

namespace sardine
{

    void register_shm_semaphore(py::module m)
    {
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
            }, py::arg("blocking") = true)
            .def_static("create", [](int init_value) -> semaphore_t& {
                return cache::request<semaphore_t>(init_value);
            }, py::arg("init_value") = 0, py::return_value_policy::reference);
        ;

        register_url(semaphore);


        py::class_<std::binary_semaphore> bin_semaphore(m, "BinarySemaphore");

        bin_semaphore
            .def("release", [](std::binary_semaphore& sem, std::size_t count){
                sem.release(count);
            }, py::arg("count") = 1)
            .def("acquire", [](std::binary_semaphore& sem, bool blocking) -> bool {
                if (blocking) {
                    sem.acquire();
                    return true;
                }
                else
                    return sem.try_acquire();
            }, py::arg("blocking") = true)
            .def_static("create", [](int init_value) -> std::binary_semaphore& {
                return cache::request<std::binary_semaphore>(init_value);
            }, py::arg("init_value") = 0, py::return_value_policy::reference);
        ;

        register_url(bin_semaphore);


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
            .def_static("create", []() -> mutex_t& {
                return cache::request<mutex_t>();
            }, py::return_value_policy::reference);
        ;

        register_url(mutex);
    }

} // namespace sardine
