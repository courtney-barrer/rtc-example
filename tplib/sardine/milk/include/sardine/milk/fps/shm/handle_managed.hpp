#ifndef FPS_SHM_HANDLE_MANAGED_H
#define FPS_SHM_HANDLE_MANAGED_H

#include <fps/detail/type.h>

#include <emu/string.h>
#include <emu/span.h>

#include <memory>

namespace fps
{

namespace shm
{

    struct handle_t;

namespace handle
{

namespace managed
{

    using s_handle_t = std::shared_ptr<handle_t>;
    using w_handle_t = std::weak_ptr<handle_t>;

    s_handle_t open(emu::string_cref name);

    s_handle_t create(emu::string_cref name);

    s_handle_t open_or_create(emu::string_cref name);

} // namespace managed

} // namespace handle

} // namespace shm

} // namespace fps

#endif //FPS_SHM_HANDLE_MANAGED_H