#include <sardine/region_register.hpp>
#include <sardine/error.hpp>

#include <list>

namespace sardine::registry
{

    using bytes_to_url_list = std::list<bytes_to_url_t>;
    using url_to_bytes_list = std::list<std::pair<std::string, url_to_bytes_t>>;

    bytes_to_url_list& bytes_to_url_instance() {
        static bytes_to_url_list list;
        return list;
    }

    url_to_bytes_list& url_to_bytes_instance() {
        static url_to_bytes_list list;
        return list;
    }

    void register_url_region_converter( std::string scheme_name, bytes_to_url_t btu, url_to_bytes_t utb ) {
        bytes_to_url_instance().push_back(std::move(btu));
        url_to_bytes_instance().emplace_back(std::move(scheme_name), std::move(utb));
    }

    optional<result<url>> url_from_bytes( span_cb data ) {
        for (auto& converter : bytes_to_url_instance())
            if (auto res = converter(data); res)
                return *res;

        return nullopt;
    }

    result<bytes_and_device> url_to_bytes( url_view u ) {
        auto scheme = u.scheme();

        for (auto& [name, converter] : url_to_bytes_instance())
            if (name == scheme)
                return converter(u);

        return make_unexpected(error::url_unknown_scheme);
    }

} // namespace sardine::registry
