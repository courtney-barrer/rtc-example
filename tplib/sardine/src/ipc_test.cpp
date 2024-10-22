    #include <sardine/sardine.hpp>

    #include <emu/cuda.hpp>

    #include <boost/interprocess/managed_shared_memory.hpp>
    #include <cstdlib> //std::system
    #include <sstream>

    #define CUDACHECK(err)                                                         \
        do {                                                                         \
            cuda_check((err), __FILE__, __LINE__);                                     \
        } while (false)
    inline void cuda_check(cudaError_t error_code, const char *file, int line) {
        if (error_code != cudaSuccess) {
            fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code,
                    cudaGetErrorString(error_code), file, line);
            fflush(stderr);
            exit(error_code);
        }
    }

    int main(int argc, char *argv[]) {
    using namespace boost::interprocess;
    if (argc == 1) { // Parent process

        auto u_s = cuda::memory::device::make_unique_span<int>(10);
        auto d_span = u_s.get();

        fmt::print("ptr: {}\n", fmt::ptr(d_span.data()));

        auto url = sardine::url_of(d_span).value();

        // // Create a managed shared memory segment
        // managed_shared_memory segment(create_only, "MySharedMemory", 65536);

        // // Allocate a portion of the segment (raw memory)
        // managed_shared_memory::size_type free_memory = segment.get_free_memory();
        // void *shptr = segment.allocate(sizeof(cudaIpcMemHandle_t) /*bytes to allocate*/);

        // // Check invariant
        // if (free_memory <= segment.get_free_memory())
        //     return 1;

        // // An handle from the base address can identify any byte of the shared
        // // memory segment even if it is mapped in different base addresses
        // managed_shared_memory::handle_t handle =
        //     segment.get_handle_from_address(shptr);
        std::stringstream s;
        s << argv[0] << ' ' << url;
        s << std::endl;
        // Launch child process
        if (0 != std::system(s.str().c_str()))
            return 1;

        int wait;

        // Check memory has been freed
        // if (free_memory != segment.get_free_memory())
        // return 1;
    } else {
        // Open managed segment
        // auto managed = sardine::region::managed::open("MySharedMemory");

        // auto& handle = managed.open<cudaIpcMemHandle_t>("h");
        // int *dp;
        // CUDACHECK(cudaIpcOpenMemHandle((void**)&dp, handle, cudaIpcMemLazyEnablePeerAccess));


        // fmt::print("ptr: {}\n", fmt::ptr(dp));

        // CUDACHECK(cudaIpcCloseMemHandle((void*)dp));
        // managed_shared_memory segment(open_only, "MySharedMemory");

        // // An handle from the base address can identify any byte of the shared
        // // memory segment even if it is mapped in different base addresses
        // managed_shared_memory::handle_t handle = 0;

        std::string u_s;

        // // Obtain handle value
        std::stringstream s;
        s << argv[1];
        s >> u_s;

        sardine::url u(u_s);

        auto d_span2 = sardine::from_url<std::span<int>>(u).value();

        fmt::print("ptr: {}\n", fmt::ptr(d_span2.data()));


        // // Get buffer local address from handle
        // void *msg = segment.get_address_from_handle(handle);

        // // Deallocate previously allocated memory
        // segment.deallocate(msg);
    }
    return 0;
    }
