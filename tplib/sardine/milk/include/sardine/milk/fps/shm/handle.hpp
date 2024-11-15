#ifndef FPS_HANDLE_H
#define FPS_HANDLE_H

#include <fps/detail/type.h>
#include <fps/detail/utility.h>

#include <emu/string.h>
#include <emu/scoped.h>
#include <emu/optional.h>

#include <unordered_map>
#include <cstring>
#include <vector>
#include <cstdint>

namespace fps
{

namespace shm
{

    struct handle_t;

    /// Convert a fps reference name to a absolute filename.
    std::string filename(emu::string_cref name);

    /// True if the fps reference name points to an existing fps, else False.
    bool exists(emu::string_cref name);

    /// Remove the fps file.
    void remove(emu::string_cref name);

    void throw_already_exists(emu::string_cref name);

    void throw_if_exists(emu::string_cref name);

    struct remover {
        std::string name;
        remover(std::string name);
        ~remover();
    };

namespace handle
{

    using id_t = FUNCTION_PARAMETER_STRUCT;

    using parameter_t = FUNCTION_PARAMETER;

namespace detail
{

    /// Open an existing fps.
    id_t open(emu::string_cref name);

    /// Create a fps and connect to it.
    id_t create(emu::string_cref name);

    /// Create a or connect to a fps.
    id_t open_or_create(emu::string_cref name, bool create_handle);

    void close(id_t & handle);

    void destroy(id_t & handle);

    struct Destroyer{
        void operator()(id_t& handle) const { close(handle); }
    };

    std::size_t level(key_t key);

} // namespace detail

    using ScopedHandle = emu::scoped_t<id_t, detail::Destroyer>;

} // namespace handle

    struct handle_t {

        handle_t() = default;

        handle_t(handle::id_t handle, bool owning);

        handle_t(handle_t&&) = default;
        handle_t& operator=(handle_t&&) = default;

        ~handle_t() = default;

        handle::id_t & id() noexcept {
            return id_.value;
        }

        const handle::id_t & id() const noexcept {
            return id_.value;
        }

        int parameter_index(key_t key);

        bool has_parameter(key_t key);

        shm::handle::parameter_t& init(key_t key, emu::string_cref desc, type_t type);

        shm::handle::parameter_t& parameter(key_t key);

        // int next_id(emu::string_cref prefix, std::size_t level, int last_id) const;

        emu::optional_t<std::string> next_token(emu::string_cref prefix, std::size_t level, emu::optional_t<std::string> last_token) const;
        int next_id(emu::string_cref prefix, std::size_t level, int idx) const;
        std::string key_at(int id) const;

        std::string name() const;

        void destroy();

    private:
        emu::optional_t<int> index(key_t key);

        handle::ScopedHandle id_;
        std::unordered_map<std::string, int> idx_map;
    };

namespace handle
{

    handle_t open(emu::string_cref name);

    handle_t create(emu::string_cref name);

    handle_t open_or_create(emu::string_cref name);

    handle_t wrap(id_t handle, bool take_ownership);

} // namespace handle

} // namespace shm

} // namespace fps

#endif //FPS_HANDLE_H