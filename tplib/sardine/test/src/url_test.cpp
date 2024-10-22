
#include <sardine/sardine.hpp>

#include <gtest/gtest.h>

using namespace sardine;

namespace
{

    TEST(Url, ScalarTest)
    {
        int& scalar = cache::request<int>(42);

        ASSERT_EQ(scalar, 42);

        auto maybe_url = sardine::url_of(scalar);

        ASSERT_TRUE(maybe_url) << "Could not create url: " << maybe_url.error().message();

        auto maybe_scalar2 = sardine::from_url<double>(*maybe_url);

        ASSERT_TRUE(maybe_scalar2) << "Could not open url: " << maybe_scalar2.error().message() << " url: " << maybe_url.value();

        double& scalar2 = maybe_scalar2.value();

        // ASSERT_EQ(scalar2, 42);
        // ASSERT_EQ(&scalar, &scalar2);
    }

    TEST(Url, StringTest)
    {
        // using sardine::region::managed::string;

        auto str = cache::request<std::string_view>("Hello, World!");

        ASSERT_EQ(str, "Hello, World!");

        auto maybe_url = sardine::url_of(str);

        ASSERT_TRUE(maybe_url) << "Could not create url: " << maybe_url.error().message();

        auto maybe_str2 = sardine::from_url<std::string_view>(*maybe_url);

        ASSERT_TRUE(maybe_str2) << "Could not open url: " << maybe_str2.error().message() << " url: " << maybe_url.value();

        std::string_view str2 = *maybe_str2;

        ASSERT_EQ(str2, "Hello, World!");
        ASSERT_EQ(str.data(), str2.data());
    }

} // namespace
