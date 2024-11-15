#pragma once

#include <fps/shm/error.h>
#include <fps/shm/handle_managed.h>
#include <fps/shm/handle.h>
#include <fps/shm/parameter.h>
#include <fps/detail/utility.h>

#include <emu/type_traits.h>
#include <emu/string.h>
#include <emu/optional.h>

#include <boost/iterator/iterator_facade.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <type_traits>
#include <tuple>
#include <ranges>

namespace fps
{

    struct object_iterator;
    struct flat_object_iterator;

    struct Object;

namespace cpts
{

    template<typename T>
    concept object = std::is_same_v<T, Object>;

    template<typename T>
    concept object_like = object<emu::decay<T>>;

} // namespace cpts

    /**
     * @brief Object is an now owning view to a fps field.
     *
     */
    struct Object
    {

        using iterator = object_iterator;

        Object() = default;
        Object(shm::handle_t *handle_ptr, std::string key) : handle_ptr(handle_ptr), key(key) {}

        /// Assign the value to the current field.
        template <typename T>
            requires not cpts::object_like<T>
        const Object &operator=(T &&value) const
        {
            set(value);
            return *this;
        }

        /// Assign the value to the current field.
        template <typename T>
            requires not cpts::object_like<T>
        Object &operator=(T &&value)
        {
            set(value);
            return *this;
        }

        /// Return if the field type
        template <typename T>
        bool is() const
        {
            return type() == type_of<emu::RemoveCVRef<T>>;
        }

        /// Check if the field exists with the provied type.
        template <typename T>
        bool check() const
        {
            return has_value() and is<T>();
        }

        /// Check if the field exists with the provied type. Throws otherwise.
        template <typename T>
        void check_throw() const
        {
            if (not has_value())
                shm::throw_field_does_not_exist(handle().name(), key);
            if (unchecked_type() != type_of<emu::RemoveCVRef<T>>)
                shm::throw_field_type_mismatch(handle().name(), key, type(), type_of<emu::RemoveCVRef<T>>);
        }

        /// Init or update a field with the provided value and a description.
        template <typename T>
        const Object &emplace(T &&value, emu::string_cref desc = "") const
        {
            if (not has_value())
                init(desc, type_of<emu::RemoveCVRef<T>>);
            set(value);
            return *this;
        }

        /// Init or update a field with the provided value and a description.
        template <typename T>
        Object &emplace(T &&value, emu::string_cref desc = "")
        {
            if (not has_value())
                init(desc, type_of<emu::RemoveCVRef<T>>);
            set(value);
            return *this;
        }

        /// Init a field with the provided type and a description.
        template <typename T>
        const Object &init(emu::string_cref desc = "") const
        {
            if (has_value())
                shm::throw_field_already_exist(handle().name(), key);
            init(desc, type_of<emu::RemoveCVRef<T>>);
            return *this;
        }

        /// Init a field with the provided type and a description.
        template <typename T>
        Object &init(emu::string_cref desc = "")
        {
            if (has_value())
                shm::throw_field_already_exist(handle().name(), key);
            init(desc, type_of<emu::RemoveCVRef<T>>);
            return *this;
        }

        template <typename T>
        const Object &try_init(emu::string_cref desc = "") const
        {
            if (not has_value())
                init(desc, type_of<emu::RemoveCVRef<T>>);
            else if (not is<emu::RemoveCVRef<T>>())
                shm::throw_field_type_mismatch(handle().name(), key, unchecked_type(), type_of<emu::RemoveCVRef<T>>);
            return *this;
        }

        template <typename T>
        Object &try_init(emu::string_cref desc = "")
        {
            if (not has_value())
                init(desc, type_of<emu::RemoveCVRef<T>>);
            else if (not is<emu::RemoveCVRef<T>>())
                shm::throw_field_type_mismatch(handle().name(), key, unchecked_type(), type_of<emu::RemoveCVRef<T>>);
            return *this;
        }

        /// Return the value of an existing field.
        template <typename T>
        auto value() const
        {
            check_throw<T>();
            return unsafe_value<T>();
        }

        /// Return the value of an existing field.
        template <typename T>
        auto unsafe_value() const
        {
            return shm::value<T>(parameter());
        }

        /// Return a reference of an existing field.
        template <typename T>
        decltype(auto) ref() const
        {
            check_throw<T>();
            return unsafe_ref<T>();
        }

        /// Return a reference of an existing field.
        template <typename T>
        decltype(auto) unsafe_ref() const
        {
            return shm::ref<T>(parameter());
        }

        /// Return a pointer of an existing field.
        template <typename T>
        auto ptr() const
        {
            check_throw<T>();
            return unsafe_ptr<T>();
        }

        /// Return a pointer of an existing field.
        template <typename T>
        auto unsafe_ptr() const
        {
            return &shm::ref<T>(parameter());
        }

        /// Return an optional with the field value if exist or empty optional.
        template <typename T>
        auto opt() const -> emu::optional_t<decltype(value<T>())>
        {
            if (check<T>())
                return value<T>();
            else
                // Needs to return same type than above but as empty.
                // Because of `char*`, return type is not necessarily `emu::optional_t<T>`.
                return emu::nullopt;
        }

        /// Update an existing field with the new value.
        template <typename T>
        void set(T &&value) const
        {
            check_throw<emu::RemoveCVRef<T>>();

            unsafe_set(EMU_FWD(value));
        }

        /// Update an existing field with the new value without any check.
        template <typename T>
        void unsafe_set(T &&value) const
        {
            shm::set(parameter(), EMU_FWD(value));
        }

        Object operator[](key_t key) const;
        Object operator[](const char *key) const;

        /// Returns the Object at the specified field.
        Object at(key_t key) const;

        iterator begin() const;
        iterator end() const;

        // flat_object_iterator fbegin() const;

        // flat_object_iterator fend() const;

        std::size_t size() const;

        std::size_t level() const;

        /// Test if `Object` point to a value.
        bool has_value() const;

        /// Return element type if it exist.
        type_t type() const;

        /// Return key content after last separator (dot).
        std::string name() const;

        /// Return complete key.
        std::string full_name() const;

        /// `bool` cast operator. Same as `has_value`.
        operator bool() const;

        shm::handle::parameter_t &parameter() const;

        shm::handle_t &handle();
        const shm::handle_t &handle() const;

        template <
            typename UserRet = emu::use_default, typename Fn,
            typename Ret = emu::NotDefaultOr<UserRet, decltype(std::declval<Fn>()(std::declval<const Object &>().value<octopus::i32>()))>>
        auto visit(Fn &&fn) const -> emu::optional_t<Ret>
        {
            if (not has_value())
                return emu::nullopt;

            switch (type())
            {
            case type_t::i32:
                return Ret(EMU_FWD(fn)(value<octopus::i32>()));
            case type_t::i64:
                return Ret(EMU_FWD(fn)(value<octopus::i64>()));
            case type_t::f32:
                return Ret(EMU_FWD(fn)(value<octopus::f32>()));
            case type_t::f64:
                return Ret(EMU_FWD(fn)(value<octopus::f64>()));
            case type_t::str:
                return Ret(EMU_FWD(fn)(value<std::string>()));
            }

            EMU_UNREACHABLE;
        }

        friend class object_iterator;

    protected:
        shm::handle_t *handle_ptr;
        std::string key;

        // Use as a cache. Once we accessed a parameter once. We keep it for fast access.
        mutable shm::handle::parameter_t *parameter_ = nullptr;

        std::string add_suffix(key_t o_key) const;
        void init(emu::string_cref desc, type_t type) const;
        type_t unchecked_type() const;
    };

    template <std::size_t Index, typename Obj>
        requires std::is_base_of_v<fps::Object, Obj> decltype(auto)
    get(Obj &&obj)
    {
        if constexpr (Index == 0)
            return obj.name();
        if constexpr (Index == 1)
            return EMU_FWD(obj);
    }

    using shm::handle::managed::s_handle_t;

    /**
     * @brief Represent a container of a fps shared memory.
     *
     * Behave like a fps::Object but with a empty root context.
     *
     */
    struct Context : Object
    {

        Context() = default;

        Context(s_handle_t h);

        ~Context() = default;

        int status() const;
        void set_status(int status);

        /// Owning handle to the fps shm.
        s_handle_t handle_;
    };

    struct object_iterator : boost::iterator_facade<object_iterator, Object, boost::forward_traversal_tag, Object>
    {

        Object fps;
        emu::optional_t<std::string> idx;

        // int idx;
        std::size_t level;

        object_iterator() = default;
        object_iterator(Object fps, emu::optional_t<std::string> idx, std::size_t level) : fps(fps), idx(idx), level(level)
        {
        }

    private:
        friend class boost::iterator_core_access;

        void increment() { idx = fps.handle().next_token(fps.key, level, idx); }

        bool equal(const object_iterator &other) const
        {
            return idx == other.idx and level == other.level;
        }

        Object dereference() const { return fps[*idx]; }
    };

    // struct flat_object_iterator : boost::iterator_facade<flat_object_iterator, Object, boost::forward_traversal_tag, Object>
    // {

    //     Object fps;

    //     int idx;
    //     std::size_t level;

    //     flat_object_iterator() = default;
    //     flat_object_iterator(Object fps, int idx, std::size_t level):
    //         fps(fps), idx(idx), level(level)
    //     {}

    // private:
    //     friend class boost::iterator_core_access;

    //     void increment() { idx = fps.handle().next_id(fps.key, level, idx); }

    //     bool equal(const flat_object_iterator & other) const {
    //         return idx == other.idx and level == other.level;
    //     }

    //     Object dereference() const { return {fps.handle_ptr, fps.handle().key_at(idx)}; }
    // };

    /// Open an already existing fps.
    Context open(emu::string_cref name);

    /// Create a fps.
    Context create(emu::string_cref name);

    /// Open an already existing fps or create one.
    Context open_or_create(emu::string_cref name);

    // std::string to_string(const fps::Object &fps);

} // namespace fps

template <>
struct fmt::formatter<fps::Object>
{

    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin())
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const fps::Object &obj, FormatContext &ctx) const -> decltype(ctx.out())
    {
        // Only named field are printed when obj has a value and subtree.
        auto named_field = obj.has_value() and obj.size() > 0;

        // Format value to string. If it is a string, we add quotes.
        auto format_value = [](auto value)
        {
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>)
                return fmt::format("\"{}\"", value);
            else
                return fmt::to_string(value);
        };

        // Format the object. Empty object are not printed.
        std::string value = obj.visit(format_value).value_or("");

        // Format sub elements as key value pairs.
        auto format_subtree = [](auto obj) { return fmt::format("'{}': {}", obj.name(), obj); };

        // If object has subtree, format it.
        std::string subtree = fmt::to_string(fmt::join(obj | std::views::transform(format_subtree), ", "));

        // If object has value and subtree, format it as a dictionary.
        if (subtree.size() > 0)
            subtree = fmt::format("{{{}}}", subtree);

        return fmt::format_to(ctx.out(), "{}{}{}{}", named_field ? "value: " : "", value, named_field ? ", subtree: " : "", subtree);
    }
};

namespace std
{
    template <typename Obj>
        requires std::is_base_of_v<fps::Object, Obj>
    struct tuple_size<Obj> : integral_constant<size_t, 2>
    {
    };

    template <typename Obj>
        requires std::is_base_of_v<fps::Object, Obj>
    struct tuple_element<0, Obj>
    {
        using type = std::string;
    };

    template <typename Obj>
        requires std::is_base_of_v<fps::Object, Obj>
    struct tuple_element<1, Obj>
    {
        using type = fps::Object;
    };

} // namespace std
