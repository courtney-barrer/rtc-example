cmake_minimum_required(VERSION 3.27)

project(sardine-python LANGUAGES CXX)

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
find_package(sardine REQUIRED)
find_package(emu REQUIRED)

pybind11_add_module(_sardine
    src/_sardine.cpp
    src/context.cpp
    src/url.cpp
    src/managed.cpp
    # src/json.cpp
    src/semaphore.cpp
    src/region/host.cpp
    # src/region/device.cpp
    src/buffer.cpp
)

set_target_properties(_sardine PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(_sardine PUBLIC include)

target_compile_features(_sardine PUBLIC cxx_std_20)

target_link_libraries(_sardine PRIVATE sardine::sardine emu::emu)

# Install the python module in the sardine python package.
install(TARGETS _sardine LIBRARY DESTINATION sardine)
install(FILES ${CMAKE_BINARY_DIR}/libsardine.so DESTINATION sardine)
