#pragma once

#include <sardine/url.hpp>
#include <sardine/python/cast/url.hpp>

#include <pybind11/pybind11.h>

namespace sardine
{

    //TODO: change default policy to automatic.
    template<typename RequestedType = emu::use_default, typename T>
    void register_url(pybind11::class_<T> cls, pybind11::return_value_policy policy = pybind11::return_value_policy::reference) {
        using requested_type = emu::not_default_or<RequestedType, T>;
        cls
            .def_static("__from_url__", [policy](url u) -> pybind11::object {
                decltype(auto) res = EMU_UNWRAP_OR_THROW(sardine::from_url<requested_type>(u));

                if constexpr (emu::cpts::specialization_of<decltype(res), std::reference_wrapper>)
                    return pybind11::cast(res.get(), policy);
                else
                    return pybind11::cast(res, policy);

            })
            .def("__url_of__", [](const T& value) -> url {
                return sardine::url_of(value).map_error([](auto ec){
                    fmt::print("error: {}", emu::pretty(ec));
                    emu::throw_error(ec);
                }).value();
            });
    }

} // namespace sardine
