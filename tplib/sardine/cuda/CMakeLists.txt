cmake_minimum_required(VERSION 3.27)

project(sardine-cuda VERSION 0.1.0 LANGUAGES CXX)

option(CONAN_ARGS "Additional conan args" "")

# Always generate compile_commands.json. Used by clangd and other tools.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Only when building the python module
if(SKBUILD)
    set(CONAN_INSTALL_ARGS "--build=missing;--options=baldr/*:python_module=True;")
    if (CONAN_ARGS)
        set(CONAN_INSTALL_ARGS "${CONAN_INSTALL_ARGS};${CONAN_ARGS}")
    endif()

    # We install all the libs next to each other in the sardine package.
    # Let them simply link together when installed.
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

add_library(sardinecuda SHARED
    src/lib/device.cpp
    src/lib/device/memory_converter.cpp
)

target_sources(sardinecuda PUBLIC FILE_SET HEADERS BASE_DIRS include FILES
    include/sardine/cuda/device/utility.hpp
    include/sardine/cuda/device.hpp
)

target_compile_features(sardinecuda PUBLIC cxx_std_20)

target_compile_definitions(sardinecuda PUBLIC SARDINE_CUDA)

find_package(sardine REQUIRED)
find_package(emu REQUIRED)

target_link_libraries(sardinecuda PUBLIC
    emu::cuda
    sardine::sardine
)

if (NOT SKBUILD)


    install(TARGETS sardinecuda
        FILE_SET HEADERS
    )

else()

    # in this configuration, pybind11 can't find the python interpreter by itself.
    find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_sardinecuda
        src/python/_sardinecuda.cpp
    )

    set_target_properties(_sardinecuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(_sardinecuda PUBLIC include)

    target_compile_features(_sardinecuda PUBLIC cxx_std_20)

    target_link_libraries(_sardinecuda PRIVATE sardinecuda emu::python)

    # Install the python module in the sardine python package.
    install(TARGETS _sardinecuda LIBRARY DESTINATION sardine)

    # We also copy every shared libraries to the destination
    install(TARGETS sardinecuda DESTINATION sardine)
    # file(GLOB SHARED_LIBS "${CMAKE_BINARY_DIR}/[!_]*.so*")
    # install(FILES ${SHARED_LIBS} DESTINATION sardine)

endif()

########################################################################
#
# Test
#
include(CTest)
if (BUILD_TESTING)
    add_subdirectory(test)
endif()
