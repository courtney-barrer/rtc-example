cmake_minimum_required(VERSION 3.27)

project(sardine VERSION 0.1.0 LANGUAGES CXX)

# Always generate compile_commands.json. Used by clangd and other tools.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(sardine_build_cuda "Enable cuda module" OFF)
option(sardine_build_milk "Enable milk module" OFF)

# add_compile_options (-fdiagnostics-color=always -fconcepts-diagnostics-depth=6)

add_library(sardine
    src/url.cpp
    src/type/json.cpp
    src/cache.cpp
    src/type/url.cpp
    src/error.cpp
    src/region/local.cpp
    src/region/host.cpp
    src/region/host/manager.cpp
    src/region/managed.cpp
    src/region/managed/manager.cpp
    # src/region/embedded.cpp
    src/region/cuda/device.cpp

    src/buffer/ring.cpp
    src/memory_converter.cpp
)

target_sources(sardine
    PUBLIC FILE_SET HEADERS BASE_DIRS include FILES
    include/sardine/error.hpp
    include/sardine/context.hpp
    include/sardine/concepts.hpp
    include/sardine/mapping_descriptor.hpp
    include/sardine/buffer/ring.hpp
    include/sardine/buffer/base.hpp
    include/sardine/buffer/utility.hpp
    include/sardine/buffer/interface.hpp
    include/sardine/buffer/impl.hpp
    include/sardine/fwd.hpp
    include/sardine/memory_converter.hpp
    include/sardine/utility.hpp
    include/sardine/mapper.hpp
    include/sardine/mapping_descriptor
    include/sardine/mapping_descriptor/base.hpp
    include/sardine/mapping_descriptor/interface.hpp
    include/sardine/type.hpp
    include/sardine/load.hpp
    include/sardine/cache.hpp
    include/sardine/sardine.hpp
    include/sardine/semaphore.hpp
    include/sardine/url.hpp
    include/sardine/buffer.hpp
    include/sardine/region/embedded.hpp
    include/sardine/region/managed.hpp
    include/sardine/region/managed/manager.hpp
    include/sardine/region/managed/utility.hpp
    include/sardine/region/cuda/device/utility.hpp
    include/sardine/region/cuda/device.hpp
    include/sardine/region/host/manager.hpp
    include/sardine/region/host/utility.hpp
    include/sardine/region/milk/array/synchronization/detail/lock.h
    include/sardine/region/milk/array/synchronization/detail/atomic.h
    include/sardine/region/milk/array/synchronization/handle.h
    include/sardine/region/milk/array/synchronization/type.h
    include/sardine/region/milk/array/synchronization/guard.h
    include/sardine/region/milk/array/detail/utility.h
    include/sardine/region/milk/array/detail/type.h
    include/sardine/region/milk/array/shm/handle.h
    include/sardine/region/milk/array/shm/error.h
    include/sardine/region/milk/array/shm/handle_managed.h
    include/sardine/region/milk/array/cacao.h
    include/sardine/region/milk/parameter/detail/utility.hpp
    include/sardine/region/milk/parameter/detail/type.hpp
    include/sardine/region/milk/parameter/shm/error.hpp
    include/sardine/region/milk/parameter/shm/parameter.hpp
    include/sardine/region/milk/parameter/shm/handle_managed.hpp
    include/sardine/region/milk/parameter/shm/handle.hpp
    include/sardine/region/milk/parameter/fps.hpp
    include/sardine/region/local.hpp
    include/sardine/region/host.hpp
    include/sardine/type/url.hpp
    include/sardine/type/json.hpp
    include/sardine/mapper/base.hpp
    include/sardine/mapper/mapper_base.hpp
    include/sardine/json.hpp
    include/sardine/python/managed_helper.hpp
    include/sardine/python/mapper.hpp
    include/sardine/python/cast/url.hpp
    include/sardine/python/cast/json.hpp
    include/sardine/python/url_helper.hpp
)

if(sardine_build_cuda)
    target_sources(sardine PRIVATE
        src/region/cuda/device/memory_converter.cpp
    )

    target_sources(sardine PUBLIC FILE_SET HEADERS BASE_DIRS include FILES
        include/sardine/region/cuda/device/utility.hpp
        include/sardine/region/cuda/device.hpp
    )
    # Force the link of cuda_converter symbol in src/region/cuda/memory_converter.cpp
    if(NOT BUILD_SHARED_LIBS)
        target_link_options(sardine INTERFACE "-Wl,-u,sardine_cuda_converter")
    endif()

    target_compile_definitions(sardine PUBLIC SARDINE_CUDA)
endif()

target_compile_features(sardine PUBLIC cxx_std_20)

find_package(emu REQUIRED)
find_package(Boost REQUIRED)

target_link_libraries(sardine PUBLIC
    emu::emu
    Boost::boost
)

if (sardine_build_milk)
    find_package(milk REQUIRED)
    target_link_libraries(sardine PUBLIC milk::milk)
endif()

install(TARGETS sardine
    FILE_SET HEADERS
)

# add_executable(sardine_main src/main.cpp)

# target_link_libraries(sardine_main PUBLIC sardine)

########################################################################
#
# Test
#
include(CTest)
if (BUILD_TESTING)
    add_subdirectory(test)
endif()
