#include <fps/shm/handle.hpp>

#include <emu/math.hpp>

#include <fmt/core.h>
#include <boost/algorithm/string.hpp>

#include <chrono>
#include <thread>
#include <functional>

namespace fps
{

namespace shm
{

    std::string filename(emu::string_cref name)
    {
        char shmdname[200];
        // No need for error handling, function abort program in case of error...
        function_parameter_struct_shmdirname(shmdname);

        return fmt::format("{}/{}.fps.shm", shmdname, name);
    }

    bool exists(emu::string_cref name)
    {
        return octopus::file_exists(filename(name));
    }

    void remove(emu::string_cref name)
    {
        // fmt::print("removing shm at {}\n", name);
        octopus::remove_file(filename(name));
    }

    void throw_does_not_exists(emu::string_cref name)
    {
        throw std::runtime_error(fmt::format("FPS {} does not exists", name));
    }

    void throw_already_exists(emu::string_cref name)
    {
        throw std::runtime_error(fmt::format("FPS {} already exists", name));
    }

    void throw_if_exists(emu::string_cref name)
    {
        // fmt::print("Did {} exists ? {}\n", name, exists(name));
        if (exists(name))
            throw_already_exists(name);
    }

    remover::remover(std::string name):name(name)
    {
        shm::remove(name);
    }

    remover::~remover()
    {
        shm::remove(name);
    }

namespace handle
{

namespace detail
{

    id_t open(emu::string_cref name)
    {
        using namespace std::chrono_literals;

        id_t handle;

        // `function_parameter_struct_connect` needs `SMfd` to be -1...
        handle.SMfd = -1;

        auto res = function_parameter_struct_connect(name.c_str(), &handle, FPSCONNECT_RUN);
        if (res == -1) throw_does_not_exists(name);

        return handle;
    }

    id_t create(emu::string_cref name)
    {

        throw_if_exists(name);

        function_parameter_struct_create(1024, name.c_str());

        return open(name);
    }

    id_t open_or_create(emu::string_cref name)
    {
        if (exists(name))
            return open(name);
        else
            return create(name);
    }

    void close(id_t& handle)
    {
        function_parameter_struct_disconnect(&handle);
    }

    void destroy(id_t& handle)
    {
        functionparameter_FPSremove(&handle);
    }

    bool has_value(const id_t& handle, std::size_t idx)
    {
        auto v = handle.parray[idx].fpflag;
        return v & FPFLAG_ACTIVE;
    }


    /// Return depth of provided key.
    std::size_t level(emu::string_cref key)
    {
        if (key.size() == 0)
            return 0;
        return std::count(key.begin(), key.end(), '.') + 1;
    }

    /**
     * @brief Extract the token in key at the specific level.
     *
     * The level zero is considered `root` and is empty.
     *
     * token_at("aaa.bbb.ccc", 0); // Return ""
     * token_at("aaa.bbb.ccc", 1); // Return "aaa"
     * token_at("aaa.bbb.ccc", 2); // Return "bbb"
     * token_at("aaa.bbb.ccc", 3); // Return "ccc"
     */
    std::string token_at(std::string key, int level)
    {
        if (level == 0)
            return "";

        std::vector<std::string> strs;
        boost::split(strs, key, boost::is_any_of("."));
        return strs[level - 1];
    }

    /**
     * @brief Check if key start with the provided prefix and continue with a dot (.).
     *
     * token_have_prefix("a.b", "a");  // Return true.
     * token_have_prefix("a", "a");    // Return false.
     * token_have_prefix("aa.b", "a"); // Return false.
     */
    bool token_have_prefix(key_t key, emu::string_cref prefix)
    {
        // Special case: prefix is root, all keys are prefixed by root.
        if (prefix.size() == 0)
            return true;

        // Otherwise, discard all keys that are shorter equal than prefix.
        if (key.size() <= prefix.size())
            return false;

        // Discard keys that do not start
        if (not std::equal(prefix.begin(), prefix.end(), key.begin()))
            return false;

        // key must have a dot `.` just after prefix.
        if (key[prefix.size()] != '.')
            return false;

        return true;
    }

    bool have_token_at(key_t key, emu::string_cref token, std::size_t level)
    {

        // If key is smaller than token or key level is smaller than level return false.
        if (key.size() < key.size() or std::count(key.begin(), key.end(), '.') < level)
            return false;

        return token == token_at(key, level);
    }

    int index(const id_t& handle, key_t key)
    {
        // Don't use `functionparameter_GetParamIndex`. Bugged as f****...
        for(auto i = 0; i < handle.md->NBparamMAX; ++i)
            if (handle.parray[i].fpflag & FPFLAG_ACTIVE and key == handle.parray[i].keywordfull)
                return i;
        return -1;
    }

} // namespace detail

} // namespace handle

    handle_t::handle_t(handle::id_t handle, bool owning):
        id_(handle, owning),
        idx_map()
    {}

    int handle_t::parameter_index(key_t key)
    {
        auto idx = index(key);
        if (idx == -1) [[unlikely]]
            [&]() EMU_NOINLINE { throw std::out_of_range(fmt::format("fps does not contain \"{}\" key", key)); } ();
        return idx.value();
    }

    bool handle_t::has_parameter(key_t key)
    {
        return index(key).has_value();
    }

    handle::parameter_t& handle_t::init(key_t key, emu::string_cref desc, type_t type)
    {
        long idx = 0;
        // TODO test return value even if`function_parameter_add_entry` only return SUCCESS.
        function_parameter_add_entry(&id(), key.c_str(), desc.c_str(), fps_int_type(type), FPFLAG_DEFAULT_INPUT, nullptr, &idx);
        idx_map.emplace(key, idx).second;

        return id().parray[idx];
    }

    handle::parameter_t& handle_t::parameter(key_t key)
    {
        return id().parray[parameter_index(key)];
    }

    emu::optional_t<std::string> handle_t::next_token(emu::string_cref prefix, std::size_t level, emu::optional_t<std::string> last_token) const
    {

        emu::optional_t<std::string> res;

        for(std::size_t idx = 0; idx < id().md->NBparamMAX; ++idx)
        {
            // Ignore if idx does not have value.
            if (not handle::detail::has_value(id(), idx))
                continue;

            auto key = key_at(idx);

            // Ignore if key has smaller level or does not start with prefix.
            if (handle::detail::level(key) < level or not handle::detail::token_have_prefix(key, prefix))
                continue;

            auto token = handle::detail::token_at(key, level);

            // Ignore all tokens (they have already been checked).
            if(last_token and token <= *last_token)
                continue;

            // Find the smaller token in the list that comes after `last_token`.
            res = res
                .transform(std::bind_front(emu::math::min, token))
                    //[&token](auto & res){ return std::min(res, token); })
                .value_or(token);
        }

        return res;
    }

    int handle_t::next_id(emu::string_cref prefix, std::size_t level, int idx) const
    {

        for(; idx < id().md->NBparamMAX; ++idx)
        {
            // Ignore if idx does not have value.
            if (not handle::detail::has_value(id(), idx))
                continue;

            auto key = key_at(idx);

            // Ignore if key has smaller level or does not start with prefix.
            if (handle::detail::level(key) < level or not handle::detail::token_have_prefix(key, prefix))
                continue;

            return idx;

        }

        return -1;
    }


    std::string handle_t::key_at(int idx) const
    {
        return id().parray[idx].keywordfull;
    }

    emu::optional_t<int> handle_t::index(key_t key)
    {
        auto it = idx_map.find(key);

        if (it != idx_map.end())
            return it->second; // Found in cache, return it.
        else
        {
            // Call slow search function.
            auto idx = handle::detail::index(id(), key);

            if (idx != -1) // Be sure to not store invalid index.
                return idx_map.emplace(key, idx).first->second;
            else
                return emu::nullopt;
        }
    }

    std::string handle_t::name() const
    {
        return id().md->name;
    }

    void handle_t::destroy()
    {
        handle::detail::destroy(id());
        // Keep object but will not invoke destructor anymore.
        id_.reset(id_.release(), /*owning =*/ false);
    }

namespace handle
{

    handle_t open(emu::string_cref name)
    {
        return {detail::open(name), true};
    }

    handle_t create(emu::string_cref name)
    {
        return {detail::create(name), true};
    }

    handle_t open_or_create(emu::string_cref name)
    {
        return {detail::open_or_create(name), true};
    }

    handle_t wrap(id_t handle, bool take_ownership)
    {
        return {handle, take_ownership};
    }

} // namespace handle

} // namespace shm

} // namespace fps
