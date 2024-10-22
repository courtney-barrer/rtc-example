#include <gtest/gtest.h>
#include <config.hpp>

#include <sardine/managed.hpp>

struct MyStruct {
    int* ptr;
};

template<>
struct sardine::managed::spe::managed_adaptor<MyStruct> : default_managed_adaptor<MyStruct> {
    static auto create(managed_t shm, char_ptr_holder_t name, int init_value) -> MyStruct {
        auto& seg = shm.segment_manager();

        auto ptr = seg.construct<int>(name)(init_value);

        return {ptr};
    }

    static auto exist(managed_t shm, char_ptr_holder_t name) -> bool {
        auto& seg = shm.segment_manager();
        return seg.find<int>(name).first != 0;
    }

    static auto destroy(managed_t shm, char_ptr_holder_t name) -> bool {
        auto& seg = shm.segment_manager();

        return seg.destroy<int>(name);
    }

    static auto from_region(managed_t shm, span_b region) -> MyStruct {
        auto& seg = shm.segment_manager();

        auto ptr = reinterpret_cast<int*>(region.data());

        return {ptr};
    }

    static auto region_of(MyStruct value) -> span_cb {
        return std::as_bytes(std::span{value.ptr, 1});
    }

};


namespace
{

    TEST(ManagedTest, CreateManaged)
    {
        auto managed = sardine::managed::create(managed_filename, managed_filesize);

        ASSERT_EQ(managed.shm().get_size(), managed_filesize);

        auto managed_2 = sardine::managed::open(managed_filename);

        ASSERT_EQ(&managed_2.shm(), &managed.shm());
    }

    TEST(ManagedTest, ScalarTest) {
        auto managed = sardine::managed::open_or_create(managed_filename, managed_filesize);

        auto& scalar = managed.force_create<int>("scalar", 42);

        ASSERT_EQ(scalar, 42);

        auto& scalar_2 = managed.open<int>("scalar");

        ASSERT_EQ(scalar_2, 42);

        scalar_2 = 43;

        ASSERT_EQ(scalar, 43);

    }

    TEST(ManagedTest, SpanTest) {
        auto managed = sardine::managed::open_or_create(managed_filename, managed_filesize);

        auto span = managed.force_create<std::span<int>>("span", 10);

        ASSERT_EQ(span.size(), 10);

        auto span_2 = managed.open<std::span<int>>("span");

        ASSERT_EQ(span_2.size(), 10);

        span_2[0] = 42;

        ASSERT_EQ(span[0], 42);

    }

    TEST(ManagedTest, UrlTest) {
        auto managed = sardine::managed::open_or_create(managed_filename, managed_filesize);

        auto span = managed.force_create<std::span<int>>("span", 10);

        auto url = sardine::managed::url_of(span).value();

        ASSERT_EQ(url.scheme(), "managed");
        ASSERT_EQ(url.path(), "/span/40");

        auto span_2 = sardine::managed::from_url<std::span<int>>(url).value();

        ASSERT_EQ(span_2.size(), 10);

        span_2[0] = 42;

        ASSERT_EQ(span[0], 42);

        auto& scalar = span[0];

        auto url_2 = sardine::managed::url_of(scalar).value();

        ASSERT_EQ(url_2.scheme(), "managed");
        ASSERT_EQ(url_2.path(), "/span/4");

        int& scalar_2 = sardine::managed::from_url<int&>(url_2).value();

        ASSERT_EQ(scalar_2, 42);

    }

    TEST(ManagedTest, CustomTypeTest) {
        auto managed = sardine::managed::open_or_create(managed_filename, managed_filesize);

        auto my_struct = managed.force_create<MyStruct>("my_struct", 42);

        ASSERT_EQ(*my_struct.ptr, 42);

        auto my_struct_2 = managed.open<MyStruct>("my_struct");

        ASSERT_EQ(*my_struct_2.ptr, 42);

        *my_struct_2.ptr = 43;

        ASSERT_EQ(*my_struct.ptr, 43);

    }

    TEST(ManagedTest, SimpleMultiProcessTest) {
        auto managed = sardine::managed::open_or_create(managed_filename, managed_filesize);

        int& value = managed.force_create<int>("value", 42);

        EXPECT_EXIT({
            auto managed = sardine::managed::open(managed_filename);

            int& value = managed.open<int>("value");

            EXPECT_EQ(value, 42);

            value = 43;

            std::exit(0);
        }, ::testing::ExitedWithCode(0), "");

        EXPECT_EQ(value, 43);
    }

    TEST(ManagedTest, CleanDestroyTest) {
        auto managed = sardine::managed::create(managed_filename_2, managed_filesize);

        auto free_space = managed.shm().get_free_memory();

        int& value = managed.force_create<int>("value", 42);

        auto span = managed.force_create<std::span<int>>("span", 10);

        using vector_t = sardine::managed::vector<int>;

        auto& vector = managed.force_create<vector_t>("vector", 10);

        ASSERT_EQ(vector.size(), 10);

        auto handle = sardine::managed::find_handle(reinterpret_cast<const std::byte*>(vector.data()));

        ASSERT_TRUE(handle);

        EXPECT_EQ(handle->name, managed_filename_2);

        managed.destroy<int>("value");
        managed.destroy<std::span<int>>("span");
        managed.destroy<vector_t>("vector");

        EXPECT_EQ(managed.shm().get_free_memory(), free_space);

    }

} // namespace
