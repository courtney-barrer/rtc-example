#pragma once

#include <sardine/type.hpp>

#include <emu/cuda.hpp>
#include <emu/cuda/device/span.hpp>

namespace sardine::cuda
{
    namespace cu = ::cuda;

    using cu::device_t;
    using cu::stream_t;

    //TODO: considere using emu::cude::device::span instead
    using view_b = emu::span<byte>;

    struct handle_impl {
        cudaIpcMemHandle_t handle;
        size_t size;
        // int device_id;
    };

    struct handle {
        cu::memory::ipc::imported_ptr_t data;
        size_t size;
        // view_b data;
        // // int device_id;
        // bool owning = true;

        // handle(handle_impl impl, bool owning)
        //     : data(reinterpret_cast<byte*>(cu::memory::ipc::import(impl.handle)))
        //     , size(impl.size)
        //     // , device_id{impl.device_id}
        //     // , owning{owning}
        // {}

        byte* ptr() const {
            return data.get<byte>();
        }

        view_b view() const {
            return view_b{ptr(), size};
        }

        // ~handle() {
        //     // if (owning)
        //     //     cu::memory::ipc::unmap(data.data());
        // }
    };

    struct shm_handle {
        cstring_view name;
        size_t offset;
    };

} // namespace sardine::cuda
