#pragma once

#include <sardine/region/managed.hpp>
#include <sardine/cache.hpp>

#include <emu/macro.hpp>
#include <emu/assert.hpp>
#include <emu/pybind11/cast/cstring_view.hpp>

#include <pybind11/pybind11.h>

#include <boost/callable_traits/args.hpp>

namespace sardine
{

    enum class create_kind {
        create,
        force_create,
        open_or_create,
        cache
    };

    struct create_proxy {
        create_kind kind = create_kind::create;
        region::managed_t* shm;
        cstring_view name;

        template<typename T, typename... Args>
        auto create(Args&&... args) const -> decltype(auto) {
            switch (kind) {
                case create_kind::cache:
                    return cache::request<T>(EMU_FWD(args)...);
                case create_kind::create:
                    return shm->create<T>(name.c_str(), EMU_FWD(args)...);
                case create_kind::force_create:
                    return shm->force_create<T>(name.c_str(), EMU_FWD(args)...);
                case create_kind::open_or_create:
                    return shm->open_or_create<T>(name.c_str(), EMU_FWD(args)...);
            }
            EMU_UNREACHABLE;
        }
    };

namespace detail
{

    template<typename T>
    auto default_initializer(create_proxy proxy) -> decltype(auto) { return proxy.create<T>(); }

    template<typename T, typename Fn, typename... Args, typename... Extra>
    void register_creates_impl(
        pybind11::class_<T> cls,
        Fn fn, std::type_identity<std::tuple<create_proxy, Args...>>,
        pybind11::return_value_policy policy,
        const Extra &...extra
    ) {
        using namespace pybind11::literals;

        cls
            .def_static("__cache_create__",
                [fn](Args... args) -> decltype(auto) {
                    return fn(create_proxy(create_kind::cache, nullptr, ""), args...);
            }, policy, extra...)
            .def_static("__shm_create__",
                [fn](region::managed_t& shm, cstring_view name, Args... args) -> decltype(auto) {
                    return fn(create_proxy(create_kind::create, &shm, name), args...);
            }, policy, "shm"_a, "name"_a, extra...)
            .def_static("__shm_force_create__",
                [fn](region::managed_t& shm, cstring_view name, Args... args) -> decltype(auto) {
                    return fn(create_proxy(create_kind::force_create, &shm, name), args...);
            }, policy, "shm"_a, "name"_a, extra...)
            .def_static("__shm_open_or_create__",
                [fn](region::managed_t& shm, cstring_view name, Args... args) -> decltype(auto) {
                    return fn(create_proxy(create_kind::open_or_create, &shm, name), args...);
            }, policy, "shm"_a, "name"_a, extra...);
    }

    template<typename T, typename Fn, typename... Extra>
    void register_creates(
        pybind11::class_<T> cls,
        Fn fn,
        pybind11::return_value_policy policy,
        const Extra &...extra
    ) {
        register_creates_impl(cls, fn, std::type_identity<boost::callable_traits::args_t<Fn>>{}, policy, extra...);
    }

} // namespace detail

    template<typename T>
    struct manager_helper {
        pybind11::class_<T> cls;
        pybind11::return_value_policy policy;

        template<typename... Extra>
        manager_helper& add_default(const Extra &...extra) {
            detail::register_creates(cls, detail::default_initializer<T>, policy, extra...);
            return *this;
        }

        template<typename Init, typename... Extra>
        manager_helper& add_init(
            Init&& initializer,
            const Extra &...extra
        ) {
            detail::register_creates(cls, initializer, policy, extra...);
            return *this;
        }
    };

    template<typename T>
    manager_helper<T> register_managed(
        pybind11::class_<T> cls,
        pybind11::return_value_policy policy = pybind11::return_value_policy::reference
    ) {

        cls
            .def_static("__shm_open__", [](region::managed_t& shm, cstring_view name) -> decltype(auto) {
                return shm.open<T>(name.c_str());
            }, policy)
            .def_static("__shm_exist__", [](region::managed_t& shm, cstring_view name) -> bool {
                return shm.exist<T>(name.c_str());
            })
            .def_static("__shm_destroy__", [](region::managed_t& shm, cstring_view name) -> void {
                shm.destroy<T>(name.c_str());
            });
        ;

        return {cls, policy};
    }



} // namespace sardine

// namespace pybind11::detail
// {
//     template<>
//     struct type_caster< boost::interprocess::ipcdetail::char_ptr_holder<char> > {

//         using Value = boost::interprocess::ipcdetail::char_ptr_holder<char>;                                                      \
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
