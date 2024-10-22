
#include <sardine/sardine.hpp>

#include <gtest/gtest.h>

using namespace sardine;

namespace
{

    TEST(Json, Scalar)
    {
        int& scalar = cache::request<int>(42);

        ASSERT_EQ(scalar, 42);

        auto maybe_url = sardine::url_of(scalar);

        ASSERT_TRUE(maybe_url) << "Could not create url: " << maybe_url.error().message();

        auto value = sardine::json::value_from(*maybe_url);

        auto maybe_scalar2 = sardine::from_json<int>(value);

        ASSERT_TRUE(maybe_scalar2) << "Could not open json: " << maybe_scalar2.error().message() << " url: " << maybe_url.value();

        int& scalar2 = maybe_scalar2.value();

        ASSERT_EQ(scalar2, 42);
    }

} // namespace
