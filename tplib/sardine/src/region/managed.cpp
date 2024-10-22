#include <sardine/region/managed.hpp>
#include <sardine/error.hpp>

#include <sardine/region/managed/manager.hpp>

#include <string>
#include <charconv>

//Note: Was using the '+' before but it seems to impossible to uncode it into a myboost::url.
// Try again later or continue to use '@' in the meantime.

namespace sardine::region
{

    managed::shared_memory& managed_t::shm() const {
        return *shm_;
    }

    managed::segment_manager& managed_t::segment_manager() const {
        return *shm_->get_segment_manager();
    }

    managed::named_range managed_t::named() const {
        return managed::named_range(shm_);
    }

namespace managed
{

    managed_t open(std::string name) {
        return manager::instance().open(name.c_str());
    }

    managed_t create(std::string name, std::size_t file_size) {
        return manager::instance().create(name.c_str(), file_size);
    }

    managed_t open_or_create(std::string name, std::size_t file_size) {
        return manager::instance().open_or_create(name.c_str(), file_size);
    }

    emu::optional<shm_handle> find_handle(const byte* ptr) {
        return manager::instance().find_handle(ptr);
    }

    optional<result<url>> url_from_bytes(span_cb region) {
        return find_handle(region.data()).map([&](auto handle) -> result<url> {
            auto named = handle.shm.named();

            if (auto it = std::ranges::find(named, region.data(), &named_value_t::value); it != named.end())
                return sardine::url(fmt::format("{}://{}/{}/{}", url_scheme, handle.name, it->name(), region.size()));
            else
                return sardine::url(fmt::format("{}://{}/@{}/{}", url_scheme, handle.name, handle.offset, region.size()));
        });
    }

    // result<managed_area> from_url(url_view url) {
    result<bytes_and_device> bytes_from_url(url_view url) {
        auto shm = open(url.host());

        auto segments = url.segments();
        EMU_TRUE_OR_RETURN_UN_EC(segments.size() == 2, error::managed_invalid_url_segment_count);

        auto seg = segments.begin();

        auto position = *seg;

        byte* ptr;

        if (position[0] == '@') {
            std::size_t offset;
            // 1 is the position of the first digit after the '@' sign.
            auto [p, ec] = std::from_chars(position.data() + 1, position.data() + position.size(), offset);
            EMU_TRUE_OR_RETURN_UN_EC(ec == std::errc(), ec);

            ptr = &shm.from_offset<byte>(offset);
        } else
            ptr = &shm.open<byte>(position.c_str());

        seg++;

        position = *seg;

        std::size_t size;
        auto [p, ec] = std::from_chars(position.data(), position.data() + position.size(), size);
        EMU_TRUE_OR_RETURN_UN_EC(ec == std::errc(), ec);

        return bytes_and_device{
            .region=span_b(
                reinterpret_cast<byte*>( shm.shm().get_address() ),
                shm.shm().get_size()
            ),
            .data=span_b(ptr, size),
            .device={
                .device_type=emu::dlpack::device_type_t::kDLCPU,
                .device_id=0
            }
        };

    }

} // namespace managed

} // namespace sardine::region
