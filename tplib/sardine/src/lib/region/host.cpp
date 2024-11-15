#include <optional>
#include <sardine/region/host.hpp>
#include <sardine/region/host/manager.hpp>
#include <sardine/error.hpp>

#include <charconv>

#include <emu/pointer.hpp>

namespace sardine::region::host
{

    span_b open(string name) {
        return manager::instance().open(std::move(name));
    }

    span_b create(string name, size_t file_size) {
        return manager::instance().create(std::move(name), file_size);
    }

    span_b open_or_create(string name, size_t file_size) {
        return manager::instance().open_or_create(std::move(name), file_size);
    }

    // void register_shm(string name, span_cb region) {

    // }

    optional<shm_handle> find_handle(const byte* ptr) {
        return manager::instance().find_handle(ptr);
    }

    constexpr auto shm_path_prefix = "/dev/shm";

    // optional<shm_handle> deduce_handle(const byte* ptr) {
    //     auto desc = emu::pointer_descritor_of(ptr);

    //     EMU_TRUE_OR_RETURN_NULLOPT(desc.location.start_with(shm_path_prefix));



    // }

    // optional<url> url_from_unregistered_bytes(span_cb data) {
    //     return emu::nullopt;
    // }

    optional<result<url>> url_from_bytes(span_cb data) {
        return find_handle(data.data()).map([&](shm_handle handle) -> result<url> {
            return sardine::url(fmt::format("{}://{}/{}/{}", url_scheme, handle.name, handle.offset, data.size()));
        });
    }

    result<bytes_and_device> bytes_from_url(const url_view& url) {
        auto data = open(url.host());

        auto segments = url.segments();

        EMU_TRUE_OR_RETURN_UN_EC(segments.size() == 2, error::host_url_invalid_path);

        auto seg = segments.begin();

        size_t offset = 0;
        {
            auto offset_query = *seg;
            auto [p, ec] = std::from_chars(offset_query.data(), offset_query.data() + offset_query.size(), offset);
            EMU_TRUE_OR_RETURN_UN_EC(ec == std::errc(), ec);
        }
        EMU_TRUE_OR_RETURN_UN_EC(offset <= data.size(), error::host_url_offset_overflow);

        seg++;

        size_t size = 0;
        {
            auto size_query = *seg;
            auto [p, ec] = std::from_chars(size_query.data(), size_query.data() + size_query.size(), size);
            EMU_TRUE_OR_RETURN_UN_EC(ec == std::errc(), ec);
        }
        EMU_TRUE_OR_RETURN_UN_EC(offset + size <= data.size(), error::host_url_size_overflow);

        return bytes_and_device{
            .region=data,
            .data=data.subspan(offset, size),
            .device={
                .device_type=emu::dlpack::device_type_t::kDLCPU,
                .device_id=0
            }
        };
    }

} // namespace sardine::region::host
