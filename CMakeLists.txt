cmake_minimum_required(VERSION 3.27)

project(baldr VERSION 0.1.0 LANGUAGES CXX)

# Always generate compile_commands.json. Used by clangd and other tools.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(tplib/emu)
add_subdirectory(tplib/sardine)

#################
# BALDR LIBRARY #
#################

add_library(baldr
    src/baldr/baldr.cpp
    src/baldr/component/dm/fakedm.cpp
    src/baldr/component/camera/fakecam.cpp
    src/baldr/component/rtc/ben.cpp
    src/baldr/rtc.cpp
    src/baldr/dm.cpp
    src/baldr/camera.cpp
    src/baldr/utility/command.cpp

    src/baldr/component/camera/flicam.cpp
    src/baldr/component/dm/bmc.cpp
)

target_sources(baldr
    PUBLIC FILE_SET HEADERS BASE_DIRS include FILES
    include/baldr/dm.hpp
    include/baldr/utility/push_record.hpp
    include/baldr/utility/span_cast.hpp
    include/baldr/utility/span_format.hpp
    include/baldr/utility/runner.hpp
    include/baldr/utility/updatable.hpp
    include/baldr/utility/command.hpp
    include/baldr/rtc.hpp
    include/baldr/type.hpp
    include/baldr/component/camera/fakecam.hpp
    include/baldr/component/camera/flicam.hpp
    include/baldr/component/dm/fakedm.hpp
    include/baldr/component/dm/bmc.hpp
    include/baldr/component/rtc/ben.hpp
    include/baldr/baldr.hpp
    include/baldr/camera.hpp
)

target_compile_features(baldr PUBLIC cxx_std_20)

target_link_libraries(baldr PUBLIC
    sardine::sardine
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

###########
# INSTALL #
###########

install(TARGETS baldr
    FILE_SET HEADERS
)

####################
# BALDR EXECUTABLE #
####################

add_executable(baldr_main src/cli/main.cpp)

find_package(Boost REQUIRED url json program_options)

target_link_libraries(baldr_main PUBLIC
    baldr
    Boost::url
    Boost::json
    Boost::program_options
)

########
# TEST #
########

# include(CTest)
# if (BUILD_TESTING)
#     add_subdirectory(test)
# endif()
