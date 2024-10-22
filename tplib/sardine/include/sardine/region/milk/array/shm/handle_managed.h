#ifndef CACAO_SHM_HANDLE_MANAGED_H
#define CACAO_SHM_HANDLE_MANAGED_H

#include <cacao/detail/type.h>

#include <emu/string.h>
#include <emu/span.h>

#include <memory>

namespace cacao::shm
{

    struct handle_t;

namespace handle::managed
{

    using s_handle_t = std::shared_ptr<handle_t>;
    using w_handle_t = std::weak_ptr<handle_t>;

    s_handle_t open(emu::string_cref name);

    s_handle_t create(emu::string_cref name, emu::span_t<const std::size_t> shape, std::size_t buffer_nb , octopus::type_t type, int location);

    s_handle_t open_or_create(emu::string_cref name, emu::span_t<const std::size_t> shape, std::size_t buffer_nb , octopus::type_t type, int location);

} // namespace handle::managed

} // namespace cacao::shm

#endif //CACAO_SHM_HANDLE_MANAGED_H