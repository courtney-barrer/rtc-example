cmake_minimum_required(VERSION 3.27)

project(baldr VERSION 0.1.0 LANGUAGES CXX)

# Always generate compile_commands.json. Used by clangd and other tools.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    include/baldr/baldr.hpp
)

target_compile_features(baldr PUBLIC cxx_std_20)

find_package(sardine REQUIRED)

target_link_libraries(baldr PUBLIC
    sardine::sardine
)


##################
# IMPORT BMC SDK #
##################

set(DEFAULT_BMC_DIR "/opt/Boston Micromachines")

if(EXISTS "${DEFAULT_BMC_DIR}" AND IS_DIRECTORY "${DEFAULT_BMC_DIR}")
    target_include_directories(baldr PRIVATE
        "${DEFAULT_BMC_DIR}/include"
    )
    target_link_directories(baldr PUBLIC
        "${DEFAULT_BMC_DIR}/lib"
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
    target_include_directories(baldr PRIVATE
        "${DEFAULT_FLI_DIR}/include"
    )
    target_link_directories(baldr PUBLIC
        "${DEFAULT_FLI_DIR}/lib/release"
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

target_link_libraries(baldr_main PUBLIC baldr)

########
# TEST #
########

# include(CTest)
# if (BUILD_TESTING)
#     add_subdirectory(test)
# endif()
