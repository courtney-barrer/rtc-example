cmake_minimum_required(VERSION 3.18)

enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(sardinecudatest
    src/main.cpp
    src/device_shm_test.cpp
)

target_include_directories(sardinecudatest PRIVATE include)

target_link_libraries(sardinecudatest PRIVATE GTest::gtest sardinecuda)

gtest_discover_tests(sardinecudatest)
