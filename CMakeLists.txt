cmake_minimum_required(VERSION 3.18)

project(rtc LANGUAGES CXX)

############################
# Find Python and nanobind #
############################
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

#####################
# Create the module #
#####################
# Note: python and nanobind are automatically added as dependencies with `nanobind_add_module`.
nanobind_add_module(_rtc
  src/rtc.cpp
  src/push_record.cpp
)





target_link_libraries(_rtc PRIVATE BMC FliSdk)

target_include_directories(_rtc PRIVATE
  "/opt/Boston Micromachines/include"
  "/opt/FirstLightImaging/FliSdk/include"
)

target_link_directories(_rtc PRIVATE
  "/opt/Boston Micromachines/lib"
  "/opt/FirstLightImaging/FliSdk/lib/release"
)

target_include_directories(_rtc PRIVATE include)
target_compile_features(_rtc PRIVATE cxx_std_20)

# Copy the module to the rtc python package directory
install(TARGETS _rtc LIBRARY DESTINATION rtc)
