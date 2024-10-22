#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <config.hpp>

#include <sardine/url.hpp>
#include <sardine/region/host.hpp>

#include <sardine/buffer.hpp>

using namespace sardine;

namespace
{

    TEST(ProducerConsumer, HostIntegerProCon)
    {
        auto host = region::host::open_or_create<int>(host_filename, 1024);

        int& scalar = host.front();

        scalar = 42;

        ASSERT_EQ(scalar, 42);

        auto url = url_of(scalar).value();

        ASSERT_EQ(url.scheme(), region::host::url_scheme);
        ASSERT_EQ(url.host(), host_filename);
        ASSERT_EQ(url.path(), "/0/4");

        auto p_res = producer<int, host_context>::open(url);
        auto c_res = consumer<int, host_context>::open(url);

        ASSERT_TRUE(p_res) << "fail to open from " << url << "because " << p_res.error().message();
        ASSERT_TRUE(c_res) << "fail to open from " << url << "because " << p_res.error().message();

        auto p = p_res.value();
        auto c = c_res.value();

        ASSERT_EQ(p.view(), 42);
        ASSERT_EQ(c.view(), 42);

        p.view() = 43;

        ASSERT_EQ(p.view(), 43);
        ASSERT_EQ(c.view(), 43);

        c.view() = 44;

        ASSERT_EQ(p.view(), 44);
        ASSERT_EQ(c.view(), 44);

    }

    TEST(ProducerConsumer, RingIntegerProCon)
    {
        const size_t ring_size = 4;


        auto host = region::host::open_or_create<int>(host_filename, 1024).subspan(0, ring_size);
        auto& index_host = region::host::open_or_create<size_t>("index_host", 1).front();

        index_host = 0;

        ring::index idx(index_host, ring_size, ring::next_policy::check_next);

        auto ring_view = ring::make_view(host, idx);

        ASSERT_TRUE(ring_view) << "fail to create view because " << ring_view.error().message();

        auto maybe_url = url_of(*ring_view);

        ASSERT_TRUE(maybe_url) << "fail to create url because " << maybe_url.error().message();

        auto url = maybe_url.value();

        ASSERT_EQ(url.scheme(), ring::url_scheme);

        auto p_res = producer<int, host_context>::open(url);
        auto c_res = consumer<int, host_context>::open(url);

        ASSERT_TRUE(p_res);
        ASSERT_TRUE(c_res);

        auto p = p_res.value();
        auto c = c_res.value();

        host[0] = 0;
        host[1] = 1;
        host[2] = 2;
        host[3] = 3;

        host_context ctx;

        // Producer start pointing to idx 0 + 1, value 1
        ASSERT_EQ(p.view(), 1);
        // Consumer start pointing to idx 0, value 0
        ASSERT_EQ(c.view(), 0);

        p.view() = 41; // Set value at idx 0 to 41
        p.send(ctx);

        // Now producer points to idx 2, value 2
        ASSERT_EQ(p.view(), 2);
        // Consumer still points to idx 0, value 0
        ASSERT_EQ(c.view(), 0);

        c.recv(ctx);

        // Producer still points to idx 2, value 2
        ASSERT_EQ(p.view(), 2);
        // Consumer now points to idx 1, value 41
        ASSERT_EQ(c.view(), 41);

        p.view() = 42;
        c.recv(ctx); // producer did not send, so consumer should not change

        ASSERT_EQ(p.view(), 42);
        ASSERT_EQ(c.view(), 41);

        p.send(ctx);
        c.recv(ctx);

        ASSERT_EQ(p.view(), 3);
        ASSERT_EQ(c.view(), 42);

        p.revert(ctx);
        c.revert(ctx);

        ASSERT_EQ(p.view(), 42);
        ASSERT_EQ(c.view(), 41);



    }

    // TEST(ProducerConsumer, ManagedIntegerBufferProCon)
    // {
    //     auto managed = managed::open_or_create(managed_filename, managed_filesize);

    //     std::span<int> array = managed.force_create<std::span<int>>("span_buffer", 10);

    //     ASSERT_EQ(array.size(), 10);

    //     int i = 0;
    //     for (auto& e : array) e = i++;

    //     auto url = url_of(array).value();

    //     ASSERT_EQ(url.scheme(), managed::url_scheme);
    //     ASSERT_EQ(url.host(), managed_filename);
    //     ASSERT_EQ(url.path(), "/span_buffer/40");

    //     auto parameter = json_t::parse(R"(
    //         {
    //             "buffer": {
    //                 "index" : 0,
    //                 "size" : 2,
    //                 "offset": 2,
    //                 "nb" : 3,
    //                 "policy" : "next"
    //             }
    //         }
    //     )");

    //     auto p_res = producer<std::span<int>, host_context>::open(url, parameter);
    //     auto c_res = consumer<std::span<int>, host_context>::open(url, parameter);

    //     ASSERT_TRUE(p_res);
    //     ASSERT_TRUE(c_res);

    //     auto p = p_res.value();
    //     auto c = c_res.value();

    //     host_context ctx;

    //     {
    //         auto pv = p.view();
    //         auto cv = c.view();

    //         // same as parameter["buffer"]["size"]
    //         ASSERT_EQ(pv.size(), 2);
    //         ASSERT_EQ(cv.size(), 2);
    //         ASSERT_EQ(pv.data() - cv.data(), 2);

    //     }

    // }

} // namespace
