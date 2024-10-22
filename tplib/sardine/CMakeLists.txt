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
    # include/sardine/sardine.hpp
    include/sardine/type.hpp
    # include/sardine/fwd.hpp
    # include/sardine/type/url.hpp
    # include/sardine/type/json.hpp
    # include/sardine/concepts.hpp
    # include/sardine/url.hpp
    # include/sardine/error.hpp
    # include/sardine/utility.hpp
    # include/sardine/region/host.hpp
    # include/sardine/region/host/manager.hpp
    # include/sardine/region/host/utility.hpp
    # include/sardine/region/local.hpp
    # include/sardine/region/managed.hpp
    # include/sardine/region/managed/utility.hpp
    # include/sardine/buffer/base.hpp
    # include/sardine/buffer/ring.hpp
    # include/sardine/buffer/interface.hpp
    # include/sardine/buffer/impl.hpp
    # include/sardine/buffer/utility.hpp
    # include/sardine/context.hpp
    # include/sardine/buffer.hpp
    # include/sardine/python/cast/url.hpp
    # include/sardine/python/mapper.hpp
    # include/sardine/mapping/interface.hpp
    # include/sardine/mapping/base.hpp
    # include/sardine/mapper.hpp
    # include/sardine/mapper/base.hpp
    # include/sardine/mapper/mapper_base.hpp
    # include/sardine/memory_converter.hpp
    # include/sardine/cache.hpp
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
