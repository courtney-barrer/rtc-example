#include <gtest/gtest.h>

#include <sardine/sardine.hpp>

#include <emu/test/device_test.hpp>
#include <emu/cuda.hpp>
#include <emu/cuda/device/span.hpp>
#include <emu/ostream.hpp>

namespace
{

    CUDA_TEST(CudaDevice, InprocCudaUrl) {


        auto u_s = cuda::memory::device::make_unique_span<int>(10);
        auto d_span = u_s.get();

        auto maybe_url = sardine::url_of(d_span);

        ASSERT_TRUE(maybe_url) << "Could not create url: " << maybe_url.error().message();

        auto maybe_d_span2 = sardine::from_url<std::span<int>>(*maybe_url);

        ASSERT_TRUE(maybe_d_span2) << "Could not open url: " << fmt::to_string(emu::pretty(maybe_d_span2.error())) << " url: " << maybe_url.value();

        auto d_span2 = maybe_d_span2.value();

        ASSERT_EQ(d_span2.size(), 10);
        ASSERT_EQ(d_span2.data(), d_span.data());

    }

    CUDA_TEST(CudaDevice, AutomaticCudaRegisterHost) {

        // allocate
        auto u_s = std::make_unique<int[]>(10);
        std::span h_span(u_s.get(), 10);

        auto maybe_url = sardine::url_of(h_span, /*allow_local = */ true);

        ASSERT_TRUE(maybe_url) << "Could not create url: " << maybe_url.error().message();

        auto maybe_d_span2 = sardine::from_url<emu::cuda::device::span<int>>(*maybe_url);

        ASSERT_TRUE(maybe_d_span2) << "Could not open url: " << maybe_d_span2.error().message() << " url: " << maybe_url.value();

        auto d_span2 = maybe_d_span2.value();

        ASSERT_EQ(d_span2.size(), 10);

        auto type = ::cuda::memory::type_of(sardine::v_ptr_of(d_span2));
        // The only thing we should expect is that is not non_cuda anymore
        ASSERT_NE(type, ::cuda::memory::type_t::non_cuda);

        auto maybe_device_url = sardine::url_of(d_span2, /*allow_local = */ true);

        ASSERT_TRUE(maybe_device_url) << "Could not create device url: " << maybe_device_url.error().message();

        ASSERT_EQ(*maybe_url, *maybe_device_url) << "The url of the device span must point back to the original host region";



    }

}
