#include <updatable.hpp>

#include <cstdint>
#include <vector>
#include <string>

void register_updatable(nb::module_& m) {
    register_updatable_for<float>(m);
    register_updatable_for<uint16_t>(m);
    register_updatable_for<double>(m);
    register_updatable_for<int>(m);
    register_updatable_for<std::string>(m);
    register_updatable_for<std::vector<float>>(m);
    register_updatable_for<std::vector<double>>(m);
    register_updatable_for<std::vector<uint16_t>>(m);
    register_updatable_for<std::vector<int>>(m);
}
