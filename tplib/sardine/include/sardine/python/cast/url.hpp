#pragma once

#include <sardine/type/url.hpp>

#include <emu/pybind11.hpp>
#include <emu/optional.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace sardine::cast::url
{

    inline py::module_ parse_module() {
        return py::module_::import("urllib").attr("parse");
    }

    inline py::handle to_handle(sardine::url_view u) {
        auto buff = u.buffer();
        return parse_module().attr("urlparse")(py::str(buff.data(), buff.size())).inc_ref();
    }

    inline emu::optional<sardine::url> from_handle(py::handle h) {
        if (not h.is_none()) {
            py::str s;

            if      (py::isinstance(h, parse_module().attr("ParseResult")) )
                s = py::str(h.attr("geturl")());
            else if (py::isinstance<py::str>(h))
                s = py::str(h);
            else
                return emu::nullopt;

            return sardine::url(s.cast<std::string_view>());

            // auto result = boost::urls::parse_uri(static_cast<std::string>(s).c_str());

            // if (result.has_value()) {
                // return sardine::url(result.value());
            // }
        }
        return emu::nullopt;
    }

} // namespace sardine::cast::url

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

namespace detail
{

    template<>
    struct type_caster< sardine::url >
    {
        PYBIND11_TYPE_CASTER(sardine::url, const_name("url"));

        bool load(handle src, bool) {
            if (auto maybe_url = sardine::cast::url::from_handle(src); maybe_url) {
                value = std::move(*maybe_url);
                return true;
            } else
                return false;
        }

        static handle cast(sardine::url value, return_value_policy /* policy */, handle /* parent */) {
            return sardine::cast::url::to_handle(value);
        }
    };

    template<>
    struct type_caster< sardine::url_view >
    {
        PYBIND11_TYPE_CASTER(sardine::url_view, const_name("url_view"));

        bool load(handle src, bool) {
            // return sardine::cast::url::from_handle(src)
            //     .map([&](auto new_value) -> bool {
            //         value = std::move(new_value);
            //         return true;
            //     })
            //     .value_or(false);
            return false; // cannot convert from handle to url_view
        }

        static handle cast(sardine::url_view value, return_value_policy /* policy */, handle /* parent */) {
            return sardine::cast::url::to_handle(value);
        }
    };

} // namespace detail

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
