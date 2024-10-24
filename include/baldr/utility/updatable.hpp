#pragma once

#include <fmt/core.h>
#include <array>

namespace baldr
{

    /**
    * @brief A template struct representing an updatable value.
    *
    * This struct provides functionality to store and update a value of type T.
    * It maintains two copies of the value, referred to as "current" and "next".
    * The "current" value can be accessed and modified using various member functions and operators.
    * The "next" value can be updated using the `update` function.
    * The `commit` function can be used to make the "next" value the new "current" value.
    *
    * @tparam T The type of the value to be stored and updated.
    */
    template<typename T>
    struct updatable {
        std::array<T, 2> values; /**< An array to store the two copies of the value. */
        /**
        * @brief An atomic pointer to the current value.
        *
        * This pointer is atomic to allow for thread-safe access and modification of the current value.
        *
        */
        T* current_; /**< A pointer to the current value. */
        T* next_; /**< A pointer to the next value. */
        bool has_changed; /**< A flag indicating whether the value has changed. */

        /**
        * @brief Default constructor.
        *
        * Initializes the values array, sets the current and next pointers to the first element of the array,
        * and sets the has_changed flag to false.
        */
        updatable()
            : values{}
            , current_(&values[0])
            , next_(&values[1])
            , has_changed(false)
        {}

        /**
        * @brief Constructor with initial value.
        *
        * Initializes the values array with the given value, sets the current and next pointers to the first element of the array,
        * and sets the has_changed flag to false.
        *
        * @param value The initial value.
        */
        updatable(T value)
            : values{value, value}
            , current_(&values[0])
            , next_(&values[1])
            , has_changed(false)
        {}

        /// Get a reference to the current value.
        T& current() { return *current_; }

        /// Get a const reference to the current value.
        T const& current() const { return *current_; }

        /// Get a reference to the next value.
        T& next() { return *next_; }

        /// Get a const reference to the next value.
        T const& next() const { return *next_; }

        /// Get a reference to the current value.
        T& operator*() { return *current_; }

        /// Get a const reference to the current value.
        T const& operator*() const { return *current_; }

        /// Get a pointer to the current value.
        T* operator->() { return current_; }

        /// Get a const pointer to the current value.
        T const* operator->() const { return current_; }

        /**
        * @brief Update the next value.
        *
        * This function updates the next value with the given value and keep the information that a new value is available.
        *
        * @param value The new value.
        */
        void update(T value)
        {
            *next_ = value;
            has_changed = true;
        }

        /**
        * @brief Set the has_changed flag to true.
        *
        * This function is useful when the next value has been updated directly without using the `update` function.
        */
        void set_changed() { has_changed = true; }

        /**
        * @brief Commit the changes.
        *
        * This function makes the next value the new current value.
        * If the has_changed flag is true, it also swaps the current and next pointers.
        */
        void commit()
        {
            if (has_changed) {
                std::swap(current_, next_);
                has_changed = false;
            }
        }

    };

    template<typename T>
    updatable<T> tag_invoke( json::value_to_tag< updatable<T> >, json::value const& jv ) {
        return updatable<T>(value_to<T>(jv));
    }

} // namespace baldr


template<typename T, typename Char>
struct fmt::formatter<baldr::updatable<T>, Char> : fmt::formatter<T, Char>
{
    using base = fmt::formatter<T, Char>;

    template<typename FormatContext>
    auto format(const baldr::updatable<T>& exp, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "current: {}", exp.current());
    }
};
