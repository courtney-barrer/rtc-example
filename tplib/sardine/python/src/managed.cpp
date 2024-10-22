// #include <sardine/region/managed.hpp>
// #include <sardine/python/managed_helper.hpp>

// #include <emu/cstring_view.hpp>
// #include <emu/pybind11/cast/cstring_view.hpp>
// #include <emu/pybind11/cast/span.hpp>

// #include <pybind11/pybind11.h>

// #include <boost/preprocessor/stringize.hpp>
// #include <boost/callable_traits.hpp>

// #include <functional>
// #include <type_traits>

// namespace py = pybind11;
// namespace sa = sardine;

// using sa::region::managed_t;

// // py::object open_obj(managed_t& managed, emu::cstring_view name, py::object type) {
// //     if (py::hasattr(type, "__shm_open__"))
// //         return type.attr("__shm_open__")(managed, name);

// //     throw std::runtime_error("Unknown type");
// // }

// // py::object create_obj(managed_t& managed, emu::cstring_view name, py::object type, py::args args) {
// //     if (py::hasattr(type, "__shm_create__"))
// //         return type.attr("__shm_create__")(managed, name, *args);

// //     throw std::runtime_error("Unknown type");
// // }

// // py::object open_or_create_obj(managed_t& managed, emu::cstring_view name, py::object& type, py::args args) {
// //     if (py::hasattr(type, "__shm_open_or_create__"))
// //         return type.attr("__shm_open_or_create__")(managed, name, *args);

// //     throw std::runtime_error("Unknown type");
// // }

// // py::object force_create_obj(managed_t& managed, emu::cstring_view name, py::object& type, py::args args) {
// //     if (py::hasattr(type, "__shm_force_create__"))
// //         return type.attr("__shm_force_create__")(managed, name, *args);

// //     throw std::runtime_error("Unknown type");
// // }

// // py::object exist_obj(managed_t& managed, emu::cstring_view name, py::object& type) {
// //     if (py::hasattr(type, "__shm_exist__"))
// //         return type.attr("__shm_exist__")(managed, name);

// //     throw std::runtime_error("Unknown type");
// // }

// // py::object destroy_obj(managed_t& managed, emu::cstring_view name, py::object& type) {
// //     if (py::hasattr(type, "__shm_destroy__"))
// //         return type.attr("__shm_destroy__")(managed, name);

// //     throw std::runtime_error("Unknown type");
// // }

// namespace detail
// {

//     /**
//      * @brief Take a method and change the first argument `this` to another type.
//      *
//      * The new function will have exactly the same signature (it won't be polymorphic)
//      * and will be internally cast from the new type to the old type using operator *.
//      *
//      *
//      * @tparam Fn
//      * @tparam the new `this` type.
//      */
//     template <typename T, auto Met>
//     auto adapt_method() {
//         // Gets Method domain.
//         using args = boost::callable_traits::args_t<decltype(Met)>;

//         return []<typename This, typename... Args>(std::type_identity<std::tuple<This, Args...>>) {
//             // This lambda which is returned have the exact same domain than `Met` except that
//             // the first argument is `T` instead of `This`.
//             return [] (T& t, Args... args) {
//                 // Call the original method with the provided args. We suppose '*t' returns `this`.
//                 return std::invoke(Met, *t, EMU_FWD(args)...);
//             };
//         }(std::type_identity<args>{});
//     }

// } // namespace detail


// void register_shm_managed(py::module_ m)
// {
//     using namespace py::literals;

//     py::class_<sa::region::managed::named_value_t>(m, "named_value")
//         .def_property_readonly("name", &sa::region::managed::named_value_t::name);

//     py::class_<sa::region::managed::named_range>(m, "named_range")
//         .def("__len__", &sa::region::managed::named_range::size)
//         .def("__iter__", [](sa::region::managed::named_range& range) {
//                 return py::make_iterator<py::return_value_policy::reference_internal>(range);
//             }, py::keep_alive<0, 1>())
//     ;

//     py::class_<managed_t> managed(m, "managed");

//     managed
//         .def("named", [](managed_t& managed) {
//             return managed.named();
//         }, py::keep_alive<0, 1>())
//         .def("__repr__", [](managed_t& managed) {
//             auto total = managed.shm().get_size();
//             auto used = total - managed.shm().get_free_memory();
//             return fmt::format("sardine.shm.managed: used {}/{}", used, total);
//         })
//         .def("memory_available", [](managed_t& managed) { return managed.shm().get_free_memory();})
//         .def("memory_total",     [](managed_t& managed) { return managed.shm().get_size(); })
//         .def("memory_used",      [](managed_t& managed) {
//             return managed.shm().get_size() - managed.shm().get_free_memory();
//         })

//         .def("open", [](managed_t& managed, char_ptr_holder_t name, py::object data_type) -> py_array {
//             auto dtype = py::dtype::from_args(data_type);
//             auto bytes = region::host::open(name);

//             return py_array(
//                 /* dtype = */ dtype,
//                 /* shape = */ std::vector<py::ssize_t>(1, bytes.size() / dtype.itemsize() ),
//                 /* ptr = */ v_ptr_of(bytes),
//                 py::str() // dummy handle to avoid copying the data.
//             );
//         }, "name"_a, "dtype"_a)

//         .def("create", [](managed_t& managed, char_ptr_holder_t name, std::vector<py::ssize_t> shape, py::object data_type) -> py_array {
//             auto dtype = py::dtype::from_args(data_type);
//             size_t bytes_size = dtype.itemsize(); for (auto e : shape) bytes_size *= e;

//             auto bytes = region::host::create(name, bytes_size);

//             return py_array(
//                 /* dtype = */ dtype,
//                 /* shape = */ move(shape),
//                 /* ptr = */ v_ptr_of(bytes),
//                 py::str() // dummy handle to avoid copying the data.
//             );
//         }, "name"_a, "shape"_a, "dtype"_a)

//         .def("open_or_create", [](managed_t& managed, char_ptr_holder_t name, std::vector<py::ssize_t> shape, py::object data_type) -> py_array {
//             auto dtype = py::dtype::from_args(data_type);
//             size_t bytes_size = dtype.itemsize(); for (auto e : shape) bytes_size *= e;

//             auto bytes = region::host::open_or_create(name, bytes_size);

//             // In case of opening an exiting host shm.
//             EMU_TRUE_OR_THROW_ERROR(bytes_size == bytes.size(), error::host_incompatible_shape);

//             return py_array(
//                 /* dtype = */ dtype,
//                 /* shape = */ move(shape),
//                 /* ptr = */ v_ptr_of(bytes),
//                 py::str() // dummy handle to avoid copying the data.
//             );
//         }, "name"_a, "shape"_a, "dtype"_a)

//     ;

//     using types = std::tuple<
//         bool,
//         int8_t,
//         int16_t,
//         int32_t,
//         int64_t,
//         uint8_t,
//         uint16_t,
//         uint32_t,
//         uint64_t,
//         // emu::half,
//         float,
//         double
//     >;

//     emu::product<types>([&]<typename T> {
//         auto name = fmt::format("proxy_{}", emu::numeric_name<T>);

//         using span_t = std::span<T>;

//         struct proxy_scalar {
//             managed_t& managed;

//             constexpr managed_t& operator*() { return managed; }
//         };

//         py::class_<proxy_scalar>(m, name.c_str())
//             .def("open",
//                  detail::adapt_method<proxy_scalar, &managed_t::open<span_t>>(),
//                  "name"_a)
//             .def("create",
//                  detail::adapt_method<proxy_scalar, &managed_t::create<span_t, std::size_t, T>>(),
//                  "name"_a, "size"_a = std::size_t(1), "value"_a = T{})
//             .def("open_or_create",
//                  detail::adapt_method<proxy_scalar, &managed_t::open_or_create<span_t, std::size_t, T> >(),
//                  "name"_a, "size"_a = std::size_t(1), "value"_a = T{})
//             .def("force_create",
//                  detail::adapt_method<proxy_scalar, &managed_t::force_create<span_t, std::size_t, T>>(),
//                  "name"_a, "size"_a = std::size_t(1), "value"_a = T{})
//             // .def("set",
//                     // detail::adapt_method<proxy_scalar, &managed_t::set<T>>(),
//                     // "name"_a, "value"_a = T{})
//             .def("exist",
//                  detail::adapt_method<proxy_scalar, &managed_t::exist<span_t>>(),
//                  "name"_a)
//             .def("destroy",
//                  detail::adapt_method<proxy_scalar, &managed_t::destroy<span_t>>(),
//                  "name"_a)
//         ;

//         managed.def_property_readonly(name.c_str(), [](managed_t& managed){
//             return proxy_scalar{managed};
//         });

//     });

//     // {
//     //     struct proxy_bytes {
//     //         managed_t& managed;

//     //         constexpr managed_t& operator*() { return managed; }
//     //     };

//     //     py::class_<proxy_bytes>(m, "proxy_bytes")
//     //         .def("open"          , detail::adapt_method<proxy_bytes, &managed_t::open          <std::span<std::byte>             > >() , "name"_a          )
//     //         .def("create"        , detail::adapt_method<proxy_bytes, &managed_t::create        <std::span<std::byte>, std::size_t> >() , "name"_a, "size"_a)
//     //         .def("open_or_create", detail::adapt_method<proxy_bytes, &managed_t::open_or_create<std::span<std::byte>, std::size_t> >() , "name"_a, "size"_a)
//     //         .def("force_create"  , detail::adapt_method<proxy_bytes, &managed_t::force_create  <std::span<std::byte>, std::size_t> >() , "name"_a, "size"_a)
//     //         .def("exist"         , detail::adapt_method<proxy_bytes, &managed_t::exist         <std::span<std::byte>             > >() , "name"_a          )
//     //         .def("destroy"       , detail::adapt_method<proxy_bytes, &managed_t::destroy       <std::span<std::byte>             > >() , "name"_a          )
//     //     ;

//     //     managed.def_prop_ro("proxy_bytes", [](managed_t& managed){
//     //         return proxy_bytes{managed};
//     //     });
//     // }

//     m.def("open",           &sa::region::managed::open,           "name"_a                       );
//     m.def("create",         &sa::region::managed::create,         "name"_a, "initial_count"_a = 1);
//     m.def("open_or_create", &sa::region::managed::open_or_create, "name"_a, "initial_count"_a = 1);

// }




// // py::object open(managed_t& managed, emu::cstring_view name, py::object& type) {
// //     if (py::hasattr(type, "__shm_open__"))
// //         return type.attr("__shm_open__")(managed, name);


// //     // py::module_ builtins = py::module_::import("builtins");

// //     // if      (type.is(builtins.attr("int")))
// //     //     return managed.find<int>(name);
// //     // else if (py::isinstance<py::float_>(obj))
// //     //     return managed.find<>(name);
// //     // else if (py::isinstance<py::bool_>(obj))
// //     //     return managed.find<>(name);
// //     // else if (py::isinstance<py::dict>(obj))
// //     //     return managed.find<>(name);
// //     // else if (py::isinstance<py::list>(obj))
// //     //     return managed.find<>(name);
// //     // else if (py::isinstance<py::str>(obj))
// //     //     return managed.find<std::string>(name);
// //     // else
// //         throw std::runtime_error("Unknown type");
// // }
