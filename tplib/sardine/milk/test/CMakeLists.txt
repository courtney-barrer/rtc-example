cmake_minimum_required(VERSION 3.18)

enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(sardinemilktest
    src/main.cpp
    src/milk_shm_test.cpp
)

target_include_directories(sardinemilktest PRIVATE include)

target_link_libraries(sardinemilktest PRIVATE GTest::gtest sardinemilk)

gtest_discover_tests(sardinemilktest)
