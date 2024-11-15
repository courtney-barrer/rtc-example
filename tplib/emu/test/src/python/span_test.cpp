#include <gtest/gtest.h>

#include <test_utility/view.hpp>
#include <test_utility/python/core.hpp>
#include <test_utility/python/view_test.hpp>

#include <emu/span.hpp>

#include <emu/pybind11/cast/span.hpp>

namespace
{
    REGISTER_TYPED_TEST_SUITE_P(
        PythonViewTest,  // The first argument is the test case name.
        // The rest of the arguments are the test names.
        CppToPythonToCppView);

    namespace py = pybind11;

    struct span_of_int {

        using data_type = int;

        using view_type = emu::span<int>;
        using const_view_type = emu::span<const int>;

        constexpr static std::size_t rank = 1;
        constexpr static bool support_read_only = true;

        static view_type get_view() {
            // The vector stays alive for the duration of the test.
            static auto vec = emu_test::md_helper::get_vector<data_type>();
            return view_type(vec);
        }

        static const_view_type get_const_view() {
            // The vector stays alive for the duration of the test.
            static auto vec = emu_test::md_helper::get_vector<data_type>();
            return const_view_type(vec);
        }

        static py::dict get_array_interface(py::object obj) {
            return obj.attr("__array_interface__").cast<py::dict>();
        }

    };

    using SpanTestsList = testing::Types<span_of_int>;

    INSTANTIATE_TYPED_TEST_SUITE_P(SpanPythonTests,    // Instance name
                                PythonViewTest,             // Test case name
                                SpanTestsList);  // Type list

} // namespace
