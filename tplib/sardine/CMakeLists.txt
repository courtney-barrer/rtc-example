cmake_minimum_required(VERSION 3.27)

project(sardine VERSION 0.1.0 LANGUAGES CXX)

option(CONAN_ARGS "Additional conan args" "")

# Always generates compile_commands.json. Used by clangd and other tools.
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

# sardine compiles as shared to avoid singleton duplication.
add_library(sardine SHARED
    src/lib/url.cpp
    src/lib/type/json.cpp
    src/lib/cache.cpp
    src/lib/type/url.cpp
    src/lib/error.cpp
    src/lib/sink.cpp
    src/lib/region_register.cpp
    src/lib/region/local.cpp
    src/lib/region/host.cpp
    src/lib/region/host/manager.cpp
    src/lib/region/managed.cpp
    src/lib/region/managed/manager.cpp
    src/lib/buffer/ring.cpp
    src/lib/memory_converter.cpp
    src/lib/utility/sync/barrier.cpp
)

target_sources(sardine PUBLIC FILE_SET HEADERS BASE_DIRS include FILES
    include/sardine/python/dtype.hpp
    include/sardine/python/url_helper.hpp
    include/sardine/python/managed_helper.hpp
    include/sardine/python/mapper.hpp
    include/sardine/python/cast/url.hpp
    include/sardine/python/cast/json.hpp
    include/sardine/context.hpp
    include/sardine/fwd.hpp
    include/sardine/type.hpp
    include/sardine/buffer.hpp
    include/sardine/concepts.hpp
    include/sardine/utility.hpp
    include/sardine/mapper.hpp
    include/sardine/mapping_descriptor/interface.hpp
    include/sardine/mapping_descriptor/base.hpp
    include/sardine/buffer/base.hpp
    include/sardine/buffer/impl.hpp
    include/sardine/buffer/ring.hpp
    include/sardine/buffer/interface.hpp
    include/sardine/buffer/utility.hpp
    include/sardine/load.hpp
    include/sardine/url.hpp
    include/sardine/error.hpp
    include/sardine/config.hpp
    include/sardine/mapping_descriptor.hpp
    include/sardine/region_register.hpp
    include/sardine/logger.hpp
    include/sardine/cache.hpp
    include/sardine/sardine.hpp
    include/sardine/type/url.hpp
    include/sardine/type/json.hpp
    include/sardine/sink.hpp
    include/sardine/json.hpp
    include/sardine/memory_converter.hpp
    include/sardine/mapper/mapper_base.hpp
    include/sardine/mapper/base.hpp
    include/sardine/region/host/manager.hpp
    include/sardine/region/host/utility.hpp
    include/sardine/region/managed.hpp
    include/sardine/region/managed/utility.hpp
    include/sardine/region/managed/manager.hpp
    include/sardine/region/local.hpp
    include/sardine/region/embedded.hpp
    include/sardine/region/host.hpp
    include/sardine/utility/sync/sync.hpp
    include/sardine/utility/sync/spin_lock.hpp
    include/sardine/utility/sync/barrier.hpp
)

target_compile_features(sardine PUBLIC cxx_std_20)

find_package(emu REQUIRED)

target_link_libraries(sardine PUBLIC
    emu::core
)

if (NOT SKBUILD)

    # Only install the headers when building the C++ library.
    install(TARGETS sardine
        FILE_SET HEADERS
    )

else() # When building with scikit-build

    # in this configuration, pybind11 can't find the python interpreter by itself.
    find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_sardine
        src/python/_sardine.cpp
        src/python/context.cpp
        src/python/managed.cpp
        src/python/utility/sync.cpp
        src/python/region/host.cpp
        src/python/buffer.cpp
    )

    set_target_properties(_sardine PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(_sardine PUBLIC include)

    target_compile_features(_sardine PUBLIC cxx_std_20)

    target_link_libraries(_sardine PRIVATE sardine emu::python)

    # Install the python module in the sardine python package.
    install(TARGETS _sardine LIBRARY DESTINATION sardine)

    # We also copy every shared libraries to the destination
    install(TARGETS sardine DESTINATION sardine)
    file(GLOB SHARED_LIBS "${CMAKE_BINARY_DIR}/[!_]*.so*")
    install(FILES ${SHARED_LIBS} DESTINATION sardine)

endif()



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
