#pragma once

#include <sardine/config.hpp>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/container/stable_vector.hpp>
#include <boost/interprocess/containers/list.hpp>

#include <fmt/format.h>

#include <ranges>
#include <functional>

namespace sardine::region::managed
{

    using shared_memory = boost::interprocess::managed_shared_memory;
    using segment_manager = boost::interprocess::managed_shared_memory::segment_manager;

    using char_ptr_holder_t = segment_manager::char_ptr_holder_t;

    using boost::interprocess::anonymous_instance;
    using boost::interprocess::unique_instance;

    using boost::interprocess::bad_alloc;

    using handle_t = shared_memory::handle_t;

    template <typename T>
    using allocator = boost::interprocess::allocator<T, segment_manager>;

    using string = boost::interprocess::basic_string<char, std::char_traits<char>, allocator<char>>;

    template <typename T>
    using vector = boost::interprocess::vector<T, allocator<T>>;

    template <typename T>
    using stable_vector = boost::container::stable_vector<T, allocator<T>>;

    // We choose to differ from std::map and use std::less<> as default. This is because
    // we want to be able to use a different type as key than the one used in the map for lookup.
    // for instance if we want to use a string as key, we can use a string_view for lookup and avoid
    // unnecessary allocations on shm.
    template <typename Key, typename T, typename Compare = std::less<>>
    using map = boost::interprocess::map<Key, T, Compare, allocator<std::pair<const Key, T>>>;

    template <typename T>
    using list = boost::interprocess::list<T, allocator<T>>;


    struct named_range : public std::ranges::view_interface<named_range>
    {

        shared_memory* shm_;

        named_range(shared_memory* shm)
            : shm_{shm}
        {}

        auto begin() const {
            return shm_->named_begin();
        }

        auto end() const {
            return shm_->named_end();
        }

        auto size() const {
            return shm_->get_num_named_objects();
        }
    };

    using named_value_t = std::ranges::range_value_t<named_range>;

} // namespace sardine::region::managed

namespace EMU_BOOST_NAMESPACE::interprocess::ipcdetail
{

    template<typename CharT>
    const CharT* format_as(char_ptr_holder<CharT> name) {
        if (name.is_unique())
            return "unique";
        else if (name.is_anonymous())
            return "anonymous";
        else
            return name.get();
    }

} // namespace EMU_BOOST_NAMESPACE::interprocess::ipcdetail

template<typename CharT, typename Traits, typename Allocator>
struct fmt::formatter<EMU_BOOST_NAMESPACE::container::basic_string<CharT, Traits, Allocator>> : fmt::formatter<const CharT*> {
    template <typename FormatContext>
    auto format(EMU_BOOST_NAMESPACE::container::basic_string<CharT, Traits, Allocator> const &data, FormatContext &ctx) const {
        return fmt::formatter<const CharT*>::format(data.c_str(), ctx);
    }
};
