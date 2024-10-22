#include <sardine/buffer/ring.hpp>

#include <sardine/buffer.hpp>

namespace sardine::ring
{

    index::index(box<std::size_t, host_context> global_index, std::size_t buffer_nb, next_policy policy)
        : global_index(std::move(global_index))
        , idx(global_index.value)
        , buffer_nb(buffer_nb)
        , policy(policy)
    {}

    index::index(std::size_t& global_index, std::size_t buffer_nb, next_policy policy)
        : global_index(global_index)
        , idx(global_index)
        , buffer_nb(buffer_nb)
        , policy(policy)
    {}

    // result<index> open(sardine::url url, std::size_t buffer_nb, next_policy policy) {
    //     auto global_index = box<std::size_t, host_context>::open(url);
    //     if (not global_index)
    //         return unexpected(global_index.error());

    //     return index(*global_index, buffer_nb, policy);
    // }


    void index::incr_local() {
        idx = (idx + 1) % buffer_nb;
    }

    void index::decr_local() {
        idx = (idx == 0 ? buffer_nb : idx) - 1;
    }

    bool index::has_next(host_context& ctx) {
        global_index.recv(ctx);
        return global_index.value != idx;
    }

    void index::send(host_context& ctx) {
        save_index = global_index.value;
        global_index.value = idx;
        global_index.send(ctx);

        incr_local();
    }

    void index::recv(host_context& ctx) {
        // Always recv just in case.
        save_index = global_index.value;
        global_index.recv(ctx);
        if (policy == next_policy::next) { // should we check if has_next ?
            incr_local();
        } else if (policy == next_policy::last){
                idx = global_index.value;
        } else { // policy == next_policy::check_next
            if (global_index.value != idx) {
                incr_local();
            }
        }

    }

    void index::revert_send(host_context& ctx) {
        global_index.value = save_index;
        global_index.send(ctx);

        //TODO: check correctness
        decr_local();
    }

    void index::revert_recv(host_context& ctx) {
        if (policy == next_policy::next) {
            decr_local();
        } else { // policy == next_policy::last
            idx = save_index;
        }
    }

} // namespace sardine::ring
