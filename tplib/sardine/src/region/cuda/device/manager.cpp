#include <sardine/region/cuda/device/manager.hpp>

#include <sardine/region/host.hpp>
#include <sardine/utility.hpp>

#include <fmt/format.h>

namespace sardine::region::cuda::device
{


    manager &manager::instance()
    {
        static manager instance;
        return instance;
    }

    span_b manager::open(string name) {
        return find_or_emplace(static_cast<base&>(*this), name, [&name] () -> handle{
            // open the shm object.
            auto host_view = host::open(name);

            if (host_view.size() < sizeof(handle_impl)) {
                throw std::runtime_error(fmt::format("The shared memory object {} is too small to contain a cuda handle.", name));
            }

            // Get the object address in shm.
            auto& object = *reinterpret_cast<handle_impl*>(host_view.data());

            // map the cuda ipc handle into the current process.
            return {cu::memory::ipc::import(object.handle), object.size }; //handle(object, /* owning = */ true);
        })->second.view();
    }

    span_b manager::create(string name, size_t size, cu::device_t device) {
        return emplace_or_throw(static_cast<base&>(*this), name, [&] () -> handle {
            // create the shm object.
            auto host_view = host::create(name, sizeof(handle_impl));

            // allocate memory on the device locally and put it in a pool to keep it alive.
            // TODO: try to use cuda-api-wrappers ipc memory instead.
            auto buffer = cu::memory::device::make_unique_span<byte>(device, size);
            byte* ptr = buffer.data();

            pool.emplace(move(buffer));

            cu::memory::ipc::ptr_handle_t handle = cu::memory::ipc::export_(ptr);

            // Get the object address in shm and construct it.
            // No need to call the destructor when the shm object will be destroyed because handle_impl is trivially destructible.
            static_assert(std::is_trivially_destructible_v<handle_impl>, "handle_impl must be trivially destructible or manager needs to call the destructor explicitly.");
            new (reinterpret_cast<handle_impl*>(host_view.data())) handle_impl{handle, size};

            // just return the device ptr, size and device id. owning = false indicate that the handle is not responsible for unmapping the memory.
            return {cu::memory::ipc::wrap(ptr, false), size};//handle(object, /* owning = */ false);
        })->second.view();
    }

    span_b manager::at(emu::cstring_view name) {
        auto it = base::find(name);
        if (it == base::end())
            throw std::runtime_error(fmt::format("No region named {} exists.", name));
        auto& h = it->second;
        return h.view();
    }

    optional<shm_handle> manager::find_handle(const byte* ptr) {
        for (auto& [name, handle] : static_cast<base&>(*this)) {
            auto addr = handle.ptr();
            auto size = handle.size;

            if (addr <= ptr && ptr < addr + size) {
                return shm_handle{name, static_cast<size_t>(ptr - addr)};
            }
        }
        return emu::nullopt;
    }


} // namespace sardine::region::cuda::device
