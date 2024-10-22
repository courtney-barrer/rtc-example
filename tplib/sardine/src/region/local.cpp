#include <sardine/region/local.hpp>

#include <boost/interprocess/detail/os_thread_functions.hpp>

namespace sardine::local
{

    optional<result<url>> url_from_bytes(span_cb data) {
        auto pid = boost::interprocess::ipcdetail::get_current_process_id();

        return sardine::url(fmt::format("{}://{}/{}/{}", url_scheme, pid, reinterpret_cast<std::uintptr_t>(data.data()), data.size()));
    }

    result<bytes_and_device> bytes_from_url(url_view u) {
        auto pid = boost::interprocess::ipcdetail::get_current_process_id();

        if (u.host() != std::to_string(pid))
            return make_unexpected(error::local_url_invalid_host);

        auto segments = u.segments();

        if (segments.size() != 2)
            return make_unexpected(error::local_url_invalid_path);

        auto seg = segments.begin();

        auto addr = boost::lexical_cast<std::uintptr_t>(*seg);
        seg++;
        auto size = boost::lexical_cast<std::size_t>(*seg);

        span_b data{reinterpret_cast<std::byte*>(addr), size};

        return bytes_and_device{
            .region=data,
            .data=data,
            .device={
                .device_type=emu::dlpack::device_type_t::kDLCPU,
                .device_id=0
            }
        };
    }

} // namespace sardine::local
