cmake_minimum_required(VERSION 3.12.0)
project(project_nmpde LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(nmpde src/main.cpp
        src/MonodomainSolver.h
        src/MonodomainSolver.cpp)
deal_ii_setup_target(nmpde)
