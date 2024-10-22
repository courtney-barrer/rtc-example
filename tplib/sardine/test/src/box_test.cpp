#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <config.hpp>

#include <sardine/sardine.hpp>

using namespace sardine;

namespace
{

    TEST(Box, ManagedIntegerBox)
    {
        auto managed = region::managed::open_or_create(managed_filename, managed_filesize);

        host_context ctx;

        int& scalar = managed.force_create<int>("integer_box", 42);

        ASSERT_EQ(scalar, 42);

        auto url = url_of(scalar).value();

        ASSERT_EQ(url.scheme(), region::managed::url_scheme);
        ASSERT_EQ(url.host(), managed_filename);
        ASSERT_EQ(url.path(), "/integer_box/4");

        auto b = box<int>::open(url).value();

        ASSERT_EQ(b.value, 42);

        scalar = 43;

        ASSERT_EQ(b.value, 42);

        b.recv(ctx);

        ASSERT_EQ(b.value, 43);

        b.value = 44;

        ASSERT_EQ(scalar, 43);

        b.send(ctx);

        ASSERT_EQ(scalar, 44);
    }

} // namespace
