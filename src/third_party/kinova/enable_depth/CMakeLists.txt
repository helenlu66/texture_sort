cmake_minimum_required(VERSION 3.14)
project(enable_depth)

set(KORTEX_API_DIR "/home/helenlu/workspace/ros2_kortex_ws/build/kortex_api/_deps/kinova_binary_api-src")

add_executable(enable_depth enable_depth.cpp)

target_include_directories(enable_depth PRIVATE
    ${KORTEX_API_DIR}/include/client
    ${KORTEX_API_DIR}/include/client_stubs
    ${KORTEX_API_DIR}/include/messages
    ${KORTEX_API_DIR}/include/common
    ${KORTEX_API_DIR}/include
)

target_link_libraries(enable_depth
    ${KORTEX_API_DIR}/lib/release/libKortexApiCpp.a
    pthread
    protobuf
)

target_compile_features(enable_depth PRIVATE cxx_std_14)
target_compile_definitions(enable_depth PRIVATE __linux__ _OS_UNIX)
