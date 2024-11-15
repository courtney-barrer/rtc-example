#include <sardine/cuda/device.hpp>

#include <sardine/cuda/device/utility.hpp>
#include <sardine/cache.hpp>
#include <sardine/type.hpp>
#include <sardine/utility.hpp>
#include <sardine/url.hpp>
#include <sardine/region_register.hpp>

#include <boost/bimap.hpp>

#include <charconv>
#include <list>

namespace sardine::cuda::device
{

namespace detail
{

    using handle_region_map = boost::bimap< void*, handle_impl* >;

    handle_region_map& get_handle_region_map() {
        static handle_region_map instance;
        return instance;
    }

    result<handle_impl&> handle_of_region(span_cb data) {
        auto& map = get_handle_region_map();

        auto v_ptr = v_ptr_of(data);

        auto it = map.left.find(v_ptr);
        if (it == map.left.end()) {
            cudaIpcMemHandle_t ipc_handle;
            EMU_CUDA_CHECK_RETURN_UN_EC(cudaIpcGetMemHandle(&ipc_handle, v_ptr));

            handle_impl& handle = sardine::cache::request<handle_impl>(ipc_handle, data.size());

            map.insert(handle_region_map::value_type(v_ptr, &handle));

            return handle;
        }
        return *it->second;
    }

    result<span_b> region_of_handle(handle_impl& handle) {
        auto& map = get_handle_region_map();

        auto it = map.right.find(&handle);
        if (it == map.right.end()) {
            void* v_ptr;
            EMU_CUDA_CHECK_RETURN_UN_EC(cudaIpcOpenMemHandle(&v_ptr, handle.handle, cudaIpcMemLazyEnablePeerAccess));

            map.insert(handle_region_map::value_type(v_ptr, &handle));
            return span_b(b_ptr_of(v_ptr), handle.size);
        }
        return span_b(b_ptr_of(it->second), handle.size);
    }

    span_cb containing_region_of(span_cb data) {
        auto v_ptr = const_cast<void*>(reinterpret_cast<const void*>(data.data()));

        namespace cm = ::cuda::memory;

        auto pointer_descriptor = cm::pointer::wrap(v_ptr);

        return pointer_descriptor.containing_range().as_span<const byte>();
    }

    result<handle_impl&> handle_of(span_cb data) {
        auto v_ptr = v_ptr_of(data);
        namespace cm = ::cuda::memory;

        auto type = cm::type_of(v_ptr);

        if (type == cm::type_t::device_) {
            // If the device memory is not registered, we can create a new handle for it
            // then create a host shm with a generated name and return its url.
            auto region = detail::containing_region_of(data);

            return handle_of_region(region);
        }

        return make_unexpected(error::cuda_url_invalid_path);
    }


} // namespace detail

    optional<result<url>> url_from_bytes(span_cb data) {
        auto* v_ptr = v_ptr_of(data);

        // Only case where it is not an error is when memory is not device type.
        EMU_TRUE_OR_RETURN_NULLOPT(::cuda::memory::type_of(v_ptr) == ::cuda::memory::type_t::device_);

        handle_impl& handle = EMU_UNWRAP(detail::handle_of(data));

        auto region = detail::containing_region_of(data);

        return sardine::url_of(handle) // generate url from handle
            .and_then([&](url handle_url) -> result<url> { // embed handle url in cuda_device url
                auto offset = data.data() - region.data();
                auto size = data.size();

                return url()
                    .set_scheme(url_scheme)
                    .set_host(fmt::to_string(offset))
                    .set_path(fmt::to_string(size))
                    .set_params({{"handle", handle_url}});
        });
    }

    result<bytes_and_device> bytes_from_url(url_view u) {
        auto maybe_handle_url = urls::try_get_at(u.params(), "handle");
        EMU_TRUE_OR_RETURN_UN_EC(maybe_handle_url, error::cuda_url_invalid_path);

        url handle_url(*maybe_handle_url);

        return sardine::from_url<handle_impl>(handle_url) // url to cuda ipc handle
              .and_then(detail::region_of_handle) // handle to region of bytes
              .and_then([&](auto region) -> result<bytes_and_device> {

            namespace cm = ::cuda::memory;

            // Get cuda device id from addresse
            auto device = cm::pointer::wrap(v_ptr_of(region)).device();

            auto segments = u.segments();
            // We want only one segments. The offset is handle by the url's host
            EMU_TRUE_OR_RETURN_UN_EC(segments.size() == 1, error::cuda_url_invalid_path);

            size_t offset;
            {
                auto offset_query = u.host();
                auto [p, ec] = std::from_chars(offset_query.data(), offset_query.data() + offset_query.size(), offset);
                EMU_TRUE_OR_RETURN_UN_EC(ec == std::errc(), ec);
            }
            EMU_TRUE_OR_RETURN_UN_EC(offset <= region.size(), error::cuda_url_offset_overflow);

            size_t size;
            {
                auto size_query = segments.front();
                auto [p, ec] = std::from_chars(size_query.data(), size_query.data() + size_query.size(), size);
                EMU_TRUE_OR_RETURN_UN_EC(ec == std::errc(), ec);
            }
            EMU_TRUE_OR_RETURN_UN_EC(offset + size <= region.size(), error::cuda_url_size_overflow);

            return bytes_and_device{
                .region=region,
                .data=region.subspan(offset, size),
                .device={
                    .device_type=emu::dlpack::device_type_t::kDLCUDA,
                    .device_id=device.id()
                }
            };
        });

    }

} // namespace sardine::cuda::device

SARDINE_REGISTER_URL_CONVERTER(sardine_cuda_url_converter, sardine::cuda::device::url_scheme,
    sardine::cuda::device::url_from_bytes,
    sardine::cuda::device::bytes_from_url
)
