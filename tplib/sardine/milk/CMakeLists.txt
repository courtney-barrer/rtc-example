cmake_minimum_required(VERSION 3.27)

project(sardine-milk VERSION 0.1.0 LANGUAGES CXX)

option(CONAN_ARGS "Additional conan args" "")

# Always generate compile_commands.json. Used by clangd and other tools.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(sardine_build_cuda "Enable cuda module" OFF)

# Only when building the python module
if(SKBUILD)
    # check if cuda exists
    # find_package(CUDAToolkit)
    # if(CUDAToolkit_FOUND)
    #     message(STATUS "CUDA found sdlfkjsdlfkjsdASKLDJLASKJDLASKJDLASKJDLlkfjsd")
    #     set(sardine_build_cuda ON)
    # else()
    #     message(STATUS "CUDA not found sldkfjlkJLJFLSKDJFSKLDJFLSKFJSLKDJFSLKDFJ")
    # endif()

    set(CONAN_INSTALL_ARGS "--build=missing;--options=baldr/*:python_module=True;")
    if (CONAN_ARGS)
        set(CONAN_INSTALL_ARGS "${CONAN_INSTALL_ARGS};${CONAN_ARGS}")
    endif()

    # The only proper way to get the check for CUDA is to check the environment variable.
    if(DEFINED ENV{SARDINE_CUDA})
        list(APPEND CONAN_INSTALL_ARGS "--options=sardine-milk/*:cuda=True")
        set(sardine_build_cuda ON)
    endif()

    # We install all the libs next to each other in the sardine package.
    # Let them simply link together when installed.
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
else()

endif()

add_library(sardinemilk SHARED
    src/lib/error.cpp
    src/lib/array.cpp
    src/lib/array/shm/handle.cpp
)

target_sources(sardinemilk PUBLIC FILE_SET HEADERS BASE_DIRS include FILES
    include/sardine/milk/array/detail/utility.hpp
    include/sardine/milk/array/shm/handle.hpp
    include/sardine/milk/array/synchronization/detail/atomic.hpp
    include/sardine/milk/array/synchronization/detail/lock.hpp
    include/sardine/milk/array/synchronization/guard.hpp
    include/sardine/milk/array/synchronization/handle.hpp
    include/sardine/milk/array/synchronization/type.hpp
    include/sardine/milk/type.hpp
    include/sardine/milk/fps/detail/type.hpp
    include/sardine/milk/fps/detail/utility.hpp
    include/sardine/milk/fps/shm/handle_managed.hpp
    include/sardine/milk/fps/shm/handle.hpp
    include/sardine/milk/fps/shm/parameter.hpp
    include/sardine/milk/fps/fps.hpp
    include/sardine/milk/error.hpp
    include/sardine/milk/milk.hpp
    include/sardine/milk/utility.hpp
    include/sardine/milk/array.hpp
)

target_compile_features(sardinemilk PUBLIC cxx_std_20)

if(sardine_build_cuda)

    target_compile_definitions(sardinemilk PUBLIC SARDINE_CUDA)

endif()

find_package(sardine REQUIRED)
find_package(milk REQUIRED)

target_link_libraries(sardinemilk PUBLIC
    sardine::sardine
    milk::milk
)

if (NOT SKBUILD)


    install(TARGETS sardinemilk
        FILE_SET HEADERS
    )

else()

    # in this configuration, pybind11 can't find the python interpreter by itself.
    find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_sardinemilk
        src/python/_sardinemilk.cpp
        src/python/array.cpp
    )

    set_target_properties(_sardinemilk PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(_sardinemilk PUBLIC include)

    target_compile_features(_sardinemilk PUBLIC cxx_std_20)

    target_link_libraries(_sardinemilk PRIVATE sardinemilk emu::python)

    # Install the python module in the sardine python package.
    install(TARGETS _sardinemilk LIBRARY DESTINATION sardine)
    install(TARGETS sardinemilk DESTINATION sardine)

    file(GLOB SHARED_LIBS "${CMAKE_BINARY_DIR}/[!_]*.so*")
    install(FILES ${SHARED_LIBS} DESTINATION sardine)
endif()

########################################################################
#
# Test
#
include(CTest)
if (BUILD_TESTING)
    add_subdirectory(test)
endif()
