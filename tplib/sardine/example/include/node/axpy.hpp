#ifndef EXAMPLE_NODE_AXPY_HPP
#define EXAMPLE_NODE_AXPY_HPP

#include <emu/span.h>

namespace example::node
{
    void axpy(sardine::cuda::context_t& ctx, emu::span_t<float> y, emu::span_t<const float> x, float alpha, float beta;)

    struct axpy
    {

        float alpha;
        float beta;

        axpy() : alpha(1.0f), beta(0.0f) {}
        axpy(float alpha, float beta) : alpha(alpha), beta(beta) {}

        void compute(sardine::cuda::context_t& ctx, emu::span_t<float> y, emu::span_t<const float> x) {
            axpy(ctx, y, x, alpha, beta);
        }

        void update_values(float alpha, float beta) {
            this->alpha = alpha;
            this->beta = beta;
        }
    };

} // namespace example::node

SARDINE_REGISTER(m) {

    sardine::class_<example::node::axpy>(m, "axpy")
        .def(sardine::init<>())
        .def(sardine::init<float, float>())
        .def("compute", &example::node::axpy::compute)
        .def("update_values", &example::node::axpy::update_values);
}

#endif // EXAMPLE_NODE_AXPY_HPP