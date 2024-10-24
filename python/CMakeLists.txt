cmake_minimum_required(VERSION 3.27)

project(rtc-python LANGUAGES CXX)

set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

# in this configuration, pybind11 can't find the python interpreter by itself.
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(pybind11 CONFIG REQUIRED)
find_package(baldr REQUIRED)
find_package(Boost REQUIRED COMPONENTS json url system)
find_package(fmt REQUIRED)

pybind11_add_module(_sardine
    src/sardine/_sardine.cpp
    src/sardine/context.cpp
    src/sardine/url.cpp
    src/sardine/managed.cpp
    src/sardine/semaphore.cpp
    src/sardine/region/host.cpp
    src/sardine/buffer.cpp
    # src/sardine/json.cpp
    # src/sardine/region/device.cpp
)

set_target_properties(_sardine PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_features(_sardine PUBLIC cxx_std_20)

target_link_libraries(_sardine PRIVATE
    baldr::sardine
    baldr::emu-python
    Boost::url
    Boost::json
    fmt::fmt
)

pybind11_add_module(_baldr
    src/baldr/_baldr.cpp
)

find_package(fmt REQUIRED)

set_target_properties(_baldr PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_features(_baldr PUBLIC cxx_std_20)

target_link_libraries(_baldr PRIVATE
    baldr::baldr
    Boost::url
    Boost::json
    fmt::fmt
)

##################
# IMPORT BMC SDK #
##################

set(DEFAULT_BMC_DIR "/opt/Boston Micromachines")

if(EXISTS "${DEFAULT_BMC_DIR}" AND IS_DIRECTORY "${DEFAULT_BMC_DIR}")
    target_include_directories(_baldr PRIVATE
        "${DEFAULT_BMC_DIR}/include"
    )
    target_link_directories(_baldr PUBLIC
        "${DEFAULT_BMC_DIR}/lib"
    )
    target_link_libraries(_baldr PUBLIC
        BMC
    )

    target_compile_definitions(_baldr PUBLIC BALDR_BMC)
else()
    message(STATUS "Could not find BMC. Skipping...")
endif()

##################
# IMPORT FLI SDK #
##################

set(DEFAULT_FLI_DIR "/opt/FirstLightImaging/FliSdk")

if(EXISTS "${DEFAULT_FLI_DIR}" AND IS_DIRECTORY "${DEFAULT_FLI_DIR}")
    target_include_directories(_baldr PRIVATE
        "${DEFAULT_FLI_DIR}/include"
    )
    target_link_directories(_baldr PUBLIC
        "${DEFAULT_FLI_DIR}/lib/release"
        "${DEFAULT_FLI_DIR}/3rdParty"
    )
    target_link_libraries(_baldr PUBLIC
        FliSdk
    )

    target_compile_definitions(_baldr PUBLIC BALDR_FLI)
else()
    message(STATUS "Could not find FliSdk. Skipping...")
endif()


# Install the python module in the sardine python package.
install(TARGETS _sardine LIBRARY DESTINATION baldr)
install(TARGETS _baldr LIBRARY DESTINATION baldr)

file(GLOB SHARED_LIBS "${CMAKE_BINARY_DIR}/[!_]*.so*")
install(FILES ${SHARED_LIBS} DESTINATION baldr)
