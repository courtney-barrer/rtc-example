#include <sardine/context.hpp>

#include <emu/pybind11.hpp>

namespace py = pybind11;

namespace sardine
{


    void register_context(py::module_ m) {
        py::class_<sardine::host_context>(m, "host_context")
            .def(py::init<>())
        ;

#ifdef SARDINE_CUDA
        py::class_<sardine::cuda_context, sardine::host_context>(m, "cuda_context")
            // .def(py::init<>())
        ;

#endif

    }

} // namespace sardine
