cmake_minimum_required(VERSION 3.12.0)
project(project_nmpde LANGUAGES CXX C)

include(common/cmake-common.cmake)
include_directories(
        src
)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

add_executable(nmpde src/main.cpp
        src/FESolver.cpp)
deal_ii_setup_target(nmpde)
