cmake_minimum_required(VERSION 3.18)

project(rtc LANGUAGES CXX)

message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_rtc src/rtc.cpp)
target_link_libraries(_rtc PRIVATE)

target_include_directories(_rtc PRIVATE include)

target_compile_features(_rtc PRIVATE cxx_std_20)

install(TARGETS _rtc LIBRARY DESTINATION rtc)
