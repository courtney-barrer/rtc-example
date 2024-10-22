#pragma once

#include <sardine/region/managed.hpp>

#include <emu/pybind11/cast/cstring_view.hpp>

#include <pybind11/pybind11.h>

#include <boost/callable_traits/args.hpp>

namespace sardine
{

    enum class create_kind {
        create,
        force_create,
        open_or_create,
    };

    struct create_proxy {
        create_kind kind = create_kind::create;
        region::managed_t& shm;
        emu::cstring_view name;

        template<typename T, typename... Args>
        auto create(Args&&... args) const -> decltype(auto) {
            switch (kind) {
                case create_kind::create:
                    return shm.create<T>(name.c_str(), std::forward<Args>(args)...);
                case create_kind::force_create:
                    return shm.force_create<T>(name.c_str(), std::forward<Args>(args)...);
                case create_kind::open_or_create:
                    return shm.open_or_create<T>(name.c_str(), std::forward<Args>(args)...);
            }
        }
    };

namespace detail
{

    template<typename T>
    auto default_initializer(create_proxy proxy) -> decltype(auto) { return proxy.create<T>(); }

    template<typename T, typename Fn, typename... Args>
    void register_creates_impl(
        pybind11::class_<T> cls,
        Fn fn,
        pybind11::return_value_policy policy,
        std::type_identity<std::tuple<create_proxy, Args...>>
    ) {
        cls
            .def_static("__shm_create__",
                [fn](region::managed_t& shm, emu::cstring_view name, Args... args) -> decltype(auto) {
                    return fn(create_proxy(create_kind::create, shm, name), args...);
            }, policy)
            .def_static("__shm_force_create__",
                [fn](region::managed_t& shm, emu::cstring_view name, Args... args) -> decltype(auto) {
                    return fn(create_proxy(create_kind::force_create, shm, name), args...);
            }, policy)
            .def_static("__shm_open_or_create__",
                [fn](region::managed_t& shm, emu::cstring_view name, Args... args) -> decltype(auto) {
                    return fn(create_proxy(create_kind::open_or_create, shm, name), args...);
            }, policy);
    }

    template<typename T, typename Fn>
    void register_creates(
        pybind11::class_<T> cls,
        Fn fn,
        pybind11::return_value_policy policy
    ) {
        register_creates_impl(cls, fn, policy, std::type_identity<myboost::callable_traits::args_t<Fn>>{});
    }

} // namespace detail


    template<typename T, typename Fn = decltype(detail::default_initializer<T>)>
    void register_managed(
        pybind11::class_<T> cls,
        pybind11::return_value_policy policy = pybind11::return_value_policy::reference,
        Fn&& initializer = detail::default_initializer<T>
    ) {
        detail::register_creates(cls, initializer, policy);

        cls
            .def_static("__shm_open__", [](region::managed_t& shm, std::string name) -> decltype(auto) {
                return shm.open<T>(name.c_str());
            }, policy)
            .def_static("__shm_exist__", [](region::managed_t& shm, std::string name) -> bool {
                return shm.exist<T>(name.c_str());
            })
            .def_static("__shm_destroy__", [](region::managed_t& shm, std::string name) -> void {
                shm.destroy<T>(name.c_str());
            });
        ;
    }

} // namespace sardine

// namespace pybind11::detail
// {
//     template<>
//     struct type_caster< myboost::interprocess::ipcdetail::char_ptr_holder<char> > {

//         using Value = myboost::interprocess::ipcdetail::char_ptr_holder<char>;                                                      \
//         static constexpr auto Name = const_name("char_ptr_holder");                                        \
//         template <typename T_> using Cast = movable_cast_t<T_>;

//         // char_ptr_holder cannot be default constructed, so we init it with an empty string.
//         Value value = Value("");

//         static handle from_cpp(Value *p, return_value_policy policy, cleanup_list *list) {
//             if (!p)
//                 return none().release();
//             return from_cpp(*p, policy, list);
//         }
//         explicit operator Value*() { return &value; }
//         explicit operator Value&() { return (Value &) value; }
//         explicit operator Value&&() { return (Value &&) value; }

//         bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
//             const char *str = PyUnicode_AsUTF8AndSize(src.ptr(), nullptr);
//             if (!str) {
//                 PyErr_Clear();
//                 return false;
//             }
//             value = Value(str);
//             return true;
//         }

//         static handle from_cpp(Value value, return_value_policy,
//                             cleanup_list *) noexcept {
//             return PyUnicode_FromString(value.get());
//         }
//     };
// } // namespace pybind11::detail
