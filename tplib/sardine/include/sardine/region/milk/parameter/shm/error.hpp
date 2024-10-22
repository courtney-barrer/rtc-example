#ifndef FPS_SHM_ERROR_H
#define FPS_SHM_ERROR_H

#include <fps/detail/type.h>

namespace fps
{

namespace shm
{

    void throw_field_does_not_exist(std::string_view fps_name, key_t key);

    void throw_field_already_exist(std::string_view fps_name, key_t key);

    void throw_field_type_not_handled(std::string_view fps_name, key_t key, type_t type);

    void throw_field_type_mismatch(std::string_view fps_name, key_t key, type_t type_expected, type_t type_provided);

} // namespace shm

} // namespace fps

#endif //FPS_SHM_ERROR_H