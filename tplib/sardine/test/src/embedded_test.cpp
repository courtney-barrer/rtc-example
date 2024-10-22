
#include <sardine/region/embedded.hpp>

#include <gtest/gtest.h>
#include <sardine/sardine.hpp>

using namespace sardine;

namespace
{

    TEST(Embedded, ScalarTest)
    {
        int scalar = 42;

        auto url = sardine::region::embedded::url_of(scalar);

        auto maybe_scalar2 = sardine::from_url<int>(url);

        ASSERT_TRUE(maybe_scalar2) << "Could not open url: " << maybe_scalar2.error().message() << " url: " << url;

        int& scalar2 = maybe_scalar2.value();

        ASSERT_EQ(scalar2, 42);
    }

    TEST(Embedded, SpanTest)
    {
        std::vector data{1,2,3,4,5};
        std::span span = data;

        auto url = sardine::region::embedded::url_of(span);

        auto maybe_span_2 = sardine::from_url< std::span<int> >(url);

        ASSERT_TRUE(maybe_span_2) << "Could not open url: " << maybe_span_2.error().message() << " url: " << url;

        auto span_2 = *maybe_span_2;

        ASSERT_EQ(span.size(), span_2.size());

        for (size_t i = 0; i < span.size(); ++i) {
            ASSERT_EQ(span[i], span_2[i]) << "Vectors differ at index " << i;
        }
    }

    TEST(Embedded, StringTest)
    {
        std::string_view sv = "Hi mom !";

        auto url = sardine::region::embedded::url_of(sv);

        auto maybe_sv2 = sardine::from_url<std::string_view>(url);

        ASSERT_TRUE(maybe_sv2) << "Could not open url: " << maybe_sv2.error().message() << " url: " << url;

        std::string_view sv2 = *maybe_sv2;

        ASSERT_EQ(sv2, "Hi mom !");
    }

} // namespace
