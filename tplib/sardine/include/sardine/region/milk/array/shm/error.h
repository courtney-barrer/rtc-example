#ifndef CACAO_SHM_ERROR_H
#define CACAO_SHM_ERROR_H

#include <emu/string.h>

namespace cacao
{

namespace shm
{

    using error_t = int;

    void throw_if_error(error_t err, emu::string_cref msg);

    void throw_if_errno(error_t err);

    void throw_if_errno(error_t err, emu::string_cref msg);

} // namespace shm

} // namespace cacao

#endif //CACAO_SHM_ERROR_H
