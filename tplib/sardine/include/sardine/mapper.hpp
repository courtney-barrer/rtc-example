#pragma once

#include <sardine/type.hpp>
#include <sardine/type/url.hpp>
#include <sardine/mapper/base.hpp>

namespace sardine
{

    template<typename T>
    auto mapper_from_mapping_descriptor(const interface::mapping_descriptor& md, emu::capsule&& capsule = {}) -> result< mapper<T> > {
        return mapper<T>::from_mapping_descriptor(md, std::move(capsule));
    }

    template<typename T>
    auto mapper_from(T && value) -> mapper< emu::decay<T> > {
        return mapper<emu::decay<T>>::from(EMU_FWD(value));
    }

    template<typename T>
    auto as_bytes(T && value) {
        return mapper<emu::decay<T>>::as_bytes(EMU_FWD(value));
    }

    template<typename T>
    void update_url(url& u, const mapper<T>& mapper) {
        auto descriptor = mapper.mapping_descriptor();

        update_url(u, descriptor);
    }

} // namespace sardine
