#include <fmt/core.h>
#include <fmt/ranges.h>

#include <emu/mdspan.hpp>

#include <sardine/sardine.hpp>

#include <numeric>
#include <memory_resource>

template <typename T = std::byte>
using pa = std::pmr::polymorphic_allocator<T>;

using std::span;
using emu::mdspan;

using namespace sardine;

void f0() {


    struct shm_remove {
        shm_remove() { sardine::remove("test"); }
        ~shm_remove(){ sardine::remove("test"); }
    } remover;

    std::size_t buffer_nb = 3;
    std::size_t frame_nb = 5;
    std::size_t frame_size_x = 10;
    std::size_t frame_size_y = 10;

    auto frame_size = frame_size_x * frame_size_y;

    std::size_t total_size = buffer_nb * frame_nb * frame_size_x * frame_size_y;

    auto frame_cube = region::host::open_or_create<float>("test", total_size );
    auto& counter = region::host::open_or_create<size_t>("counter", 1).front();

    // auto frame_cube = emu::as_t<float>(host_region);
    // auto& counter = emu::as_t<std::size_t>(counter_region).front();

    counter = 0;

    auto index = sardine::ring::index(counter, buffer_nb, ring::next_policy::last);

    using md_type = emu::mdspan<float, emu::d4>;
    using f_type = emu::mdspan<float, emu::d2>;

    auto mds = md_type(frame_cube.data(), buffer_nb, frame_nb, frame_size_x, frame_size_y);

    fmt::print("url: {}\n", sardine::url_of(mds));

    auto cons = sardine::ring::make_consumer<host_context>(mds, index).value();
    auto producer = sardine::make_producer<host_context>(mds);

    fmt::print("cons url: {}\n", sardine::url_of(cons));
    fmt::print("prod url: {}\n", sardine::url_of(producer));

    //    buffer 0        buffer 1          buffer 2
    //      v                 v               v
    // [f0, f1, f2, ...][f0, f1, f2, ...][f0, f1, f2, ...]

    auto f0 = cons.submdspan(0);
    auto f1 = cons.submdspan(1);
    auto f2 = cons.submdspan(2);

    auto url_f0 = sardine::url_of(f0).value();
    auto f0_bis = sardine::consumer<f_type, host_context>::open(url_f0);
    if (!f0_bis) {
        fmt::print("error: {}\n", f0_bis.error());
        return;
    }

    fmt::print("f0 url: {}\n", url_f0);
    fmt::print("f0 info: {}\n", emu::info(f0_bis.value().view()));

    // goes up to 4

    auto b = sardine::box<std::size_t, host_context>(counter);

    fmt::print("box: {}\n", b);

    host_context ctx;
    index.incr_local(); // place the write index at the first position

    fmt::print("idx: {}\n", index.idx);
    cons.recv(ctx);
    fmt::print("consumer info: {}\n", emu::info(cons.view()));
    f0.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f0.view()));
    f1.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f1.view()));
    f2.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f2.view()));

    index.send(ctx);

    fmt::print("idx: {}\n", index.idx);
    cons.recv(ctx);
    fmt::print("consumer info: {}\n", emu::info(cons.view()));
    f0.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f0.view()));
    f1.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f1.view()));
    f2.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f2.view()));

    // index.incr_local();
    index.send(ctx);

    fmt::print("idx: {}\n", index.idx);
    cons.recv(ctx);
    fmt::print("consumer info: {}\n", emu::info(cons.view()));
    f0.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f0.view()));
    f1.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f1.view()));
    f2.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f2.view()));

    // index.incr_local();
    index.send(ctx);

    fmt::print("idx: {}\n", index.idx);
    cons.recv(ctx);
    fmt::print("consumer info: {}\n", emu::info(cons.view()));
    f0.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f0.view()));
    f1.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f1.view()));
    f2.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f2.view()));

    fmt::print("revert !\n");

    index.revert_send(ctx);

    fmt::print("idx: {}\n", index.idx);
    cons.recv(ctx);
    fmt::print("consumer info: {}\n", emu::info(cons.view()));
    f0.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f0.view()));
    f1.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f1.view()));
    f2.recv(ctx);
    fmt::print("frame info: {}\n", emu::info(f2.view()));


    f0_bis.value().recv(ctx);
    fmt::print("f0 info: {}\n", emu::info(f0_bis.value().view()));


    b.recv(ctx);
    fmt::print("index: {}\n", b.value);

    fmt::print("index url: {}\n", sardine::url_of(index));

    // auto u = sardine::url_of(mds);
    // if (!u) {
    //     fmt::print("error: {}\n", u.error());
    //     return;
    // }

    // fmt::print("url: {}\n", *u);


}

int main() {
    f0();

}
