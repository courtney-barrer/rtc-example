cmake_minimum_required(VERSION 3.27)

project(baldr VERSION 0.1.0 LANGUAGES CXX)

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

#################
# BALDR LIBRARY #
#################

add_library(baldr
    src/lib/baldr.cpp
    src/lib/component/dm/fakedm.cpp
    src/lib/component/camera/fakecam.cpp
    src/lib/component/rtc/ben.cpp
    src/lib/component/rtc/fakertc.cpp
    src/lib/component/rtc.cpp
    src/lib/component/dm.cpp
    src/lib/component/camera.cpp
    src/lib/utility/command.cpp
    src/lib/component/camera/flicam.cpp
    src/lib/component/dm/bmc.cpp
    src/lib/component/base.cpp
    src/lib/utility/component_info.cpp
)

target_sources(baldr
    PUBLIC FILE_SET HEADERS BASE_DIRS include FILES
    include/baldr/component/dm.hpp
    include/baldr/utility/push_record.hpp
    include/baldr/utility/span_cast.hpp
    include/baldr/utility/span_format.hpp
    include/baldr/utility/runner.hpp
    include/baldr/utility/updatable.hpp
    include/baldr/utility/command.hpp
    include/baldr/component/rtc.hpp
    include/baldr/type.hpp
    include/baldr/component/camera/fakecam.hpp
    include/baldr/component/camera/flicam.hpp
    include/baldr/component/dm/fakedm.hpp
    include/baldr/component/dm/bmc.hpp
    include/baldr/component/rtc/ben.hpp
    include/baldr/baldr.hpp
    include/baldr/component/camera.hpp
)

target_compile_features(baldr PUBLIC cxx_std_20)

find_package(sardine REQUIRED)
find_package(Boost REQUIRED)

target_link_libraries(baldr PUBLIC
    sardine::sardine
    Boost::boost
    Boost::json
)


##################
# IMPORT BMC SDK #
##################

set(DEFAULT_BMC_DIR "/opt/Boston Micromachines")

if(EXISTS "${DEFAULT_BMC_DIR}" AND IS_DIRECTORY "${DEFAULT_BMC_DIR}")
    message(STATUS "Found BMC SDK at ${DEFAULT_BMC_DIR}")

    target_include_directories(baldr PRIVATE
        "${DEFAULT_BMC_DIR}/include"
    )
    target_link_directories(baldr PUBLIC
        "${DEFAULT_BMC_DIR}/lib"
    )
    target_link_libraries(baldr PUBLIC
        BMC
    )

    target_compile_definitions(baldr PUBLIC BALDR_BMC)
else()
    message(STATUS "Could not find BMC. Skipping...")
endif()

##################
# IMPORT FLI SDK #
##################

set(DEFAULT_FLI_DIR "/opt/FirstLightImaging/FliSdk")

if(EXISTS "${DEFAULT_FLI_DIR}" AND IS_DIRECTORY "${DEFAULT_FLI_DIR}")
    message(STATUS "Found FliSdk at ${DEFAULT_FLI_DIR}")

    target_include_directories(baldr PRIVATE
        "${DEFAULT_FLI_DIR}/include"
    )
    target_link_directories(baldr PUBLIC
        "${DEFAULT_FLI_DIR}/lib/release"
        "${DEFAULT_FLI_DIR}/3rdParty"
    )
    target_link_libraries(baldr PUBLIC
        FliSdk
    )

    target_compile_definitions(baldr PUBLIC BALDR_FLI)
else()
    message(STATUS "Could not find FliSdk. Skipping...")
endif()


####################
# BALDR EXECUTABLE #
####################

add_executable(baldr_main src/cli/main.cpp)

target_link_libraries(baldr_main PUBLIC
    baldr
)


if (NOT SKBUILD)

    ###############
    # C++ INSTALL #
    ###############

    install(TARGETS baldr
        FILE_SET HEADERS
    )

else() # When building with scikit-build

    # in this configuration, pybind11 can't find the python interpreter by itself.
    find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_baldr
        src/python/_baldr.cpp
    )

    set_target_properties(baldr PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(_baldr PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_compile_features(_baldr PUBLIC cxx_std_20)

    target_link_libraries(_baldr PRIVATE
        baldr
    )

    # Install the python module in the sardine python package.
    install(TARGETS _baldr LIBRARY DESTINATION baldr)

    # We also copy every shared libraries to the destination
    install(TARGETS baldr DESTINATION baldr)
    install(TARGETS baldr_main DESTINATION baldr/bin)
    file(GLOB SHARED_LIBS "${CMAKE_BINARY_DIR}/[!_]*.so*")
    install(FILES ${SHARED_LIBS} DESTINATION baldr)

endif()


########
# TEST #
########

# include(CTest)
# if (BUILD_TESTING)
#     add_subdirectory(test)
# endif()
