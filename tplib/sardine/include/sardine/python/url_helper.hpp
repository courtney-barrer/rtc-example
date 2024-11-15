#pragma once

#include <sardine/url.hpp>
#include <sardine/python/cast/url.hpp>

#include <pybind11/pybind11.h>

namespace sardine
{

    //TODO: change default policy to automatic.
    template<typename RequestedType = emu::use_default, typename T>
    void register_url(pybind11::class_<T> cls) {
        using requested_type = emu::not_default_or<RequestedType, T>;
        cls
            .def_static("__from_url__", [](url u) -> pybind11::object {
                decltype(auto) res = EMU_UNWRAP_RES_OR_THROW(sardine::from_url<requested_type>(u));

                if constexpr (emu::cpts::specialization_of<decltype(res), std::reference_wrapper>)
                    return pybind11::cast(res.get(), pybind11::return_value_policy::reference); // Returns reference as reference.
                else
                    return pybind11::cast(res); // default policy should work here.

            })
            .def("__url_of__", [](const T& value) -> url {
                return sardine::url_of(value).value();
            });
    }

} // namespace sardine
