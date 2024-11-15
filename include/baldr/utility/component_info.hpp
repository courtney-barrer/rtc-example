#pragma once

#include <baldr/utility/command.hpp>

#include <emu/cstring_view.hpp>
#include <fmt/format.h>
#include <sardine/config.hpp>
#include <sardine/region/managed/utility.hpp>
#include <sardine/sardine.hpp>
#include <sardine/utility/sync/sync.hpp>

#include <boost/process/v2/pid.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/atomic/ipc_atomic.hpp>

#include <chrono>
#include <scoped_allocator>
#include <tuple>
#include <utility>
#include <filesystem>

namespace baldr
{
    constexpr auto component_info_resource_name = "ba_comp_info";

    namespace managed = sardine::region::managed;

    using time_point = std::chrono::time_point<std::chrono::system_clock>;

    using pid_type = boost::process::v2::pid_type;

    void acquire_lock(emu::cstring_view name);

    void release_lock(std::string_view name);

    struct ComponentInfo
    {


        using allocator_type = managed::allocator<char>;

        managed::string name_;
        managed::string f_lock_name;
        // boost::interprocess::file_lock f_lock;

        pid_type pid;
        time_point creation_time;

        boost::ipc_atomic<Command> cmd_ = Command::pause;
        boost::ipc_atomic<Status> status_ = Status::none;
        boost::ipc_atomic<size_t> loop_count = 0;


        ComponentInfo(std::string_view name, std::string_view lock_file, allocator_type allocator)
            : name_(name, allocator)
            , f_lock_name(lock_file, allocator)
        {}

        ~ComponentInfo() {
        }

        void acquire() {
            pid = boost::process::v2::current_pid();
            creation_time = std::chrono::system_clock::now();

            acquire_lock(f_lock_name);

            status_ = Status::acquired;
        }

        void release() {
            pid = pid_type{}; // Does that work ?
            status_ = Status::exited;
            try {
                release_lock(f_lock_name);
            } catch (const std::exception&) {
                // Log error
            }

        }

        Command cmd() const {
            return cmd_.load();
        }

        Command wait_unpause() {
            return cmd_.wait(Command::pause);
        }

        Command wait_new_cmd(Command last_cmd) {
            return cmd_.wait(last_cmd);
        }

        void set_cmd(Command new_cmd) {
            if (cmd() != new_cmd)  {
                cmd_ = new_cmd;
                cmd_.notify_all();
            }
        }

        Status status() const {
            return status_.load();
        }

        void set_status(Status new_status) {
            if (status() != new_status) {
                status_ = new_status;
                status_.notify_all();
            }
        }

        emu::cstring_view name() const {
            return emu::cstring_view(emu::null_terminated, name_.c_str(), name_.size());
        }
    };

    struct ComponentInfoManager
    {
        using value_type = std::pair<const managed::string, ComponentInfo>;

        using allocator_type = managed::allocator<value_type>;

        managed::list<ComponentInfo> components;
        sardine::sync::mutex_t mut;

        ComponentInfoManager(allocator_type allocator)
            : components(allocator)
        {}

        ComponentInfo& create_component_info(std::string_view name);

        void clear() {
            sardine::sync::scoped_lock lock(mut);

            components.clear();
        }

    };

    static_assert(std::uses_allocator_v<ComponentInfo, typename ComponentInfoManager::allocator_type>);

} // namespace baldr

template<typename Alloc>
struct std::uses_allocator<baldr::ComponentInfoManager, Alloc> : std::true_type {};

// template<typename Alloc>
// struct std::uses_allocator<baldr::ComponentInfo, Alloc> : std::true_type {};
