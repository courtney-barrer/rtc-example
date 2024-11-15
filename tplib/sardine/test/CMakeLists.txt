cmake_minimum_required(VERSION 3.18)

enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(sardinetest
    src/main.cpp
    src/mapping_test.cpp
    src/url_test.cpp
    src/cache_test.cpp
    src/json_test.cpp
    # src/embedded_test.cpp
    # # src/managed_test.cpp
    src/box_test.cpp
    src/view_test.cpp
    src/producer_consumer_test.cpp
)

# if(sardine_build_cuda)
#     target_sources(sardinetest PRIVATE
#         src/device_shm_test.cpp
#     )
# endif()

target_include_directories(sardinetest PRIVATE include)

target_link_libraries(sardinetest PRIVATE GTest::gtest sardine)

gtest_discover_tests(sardinetest)
