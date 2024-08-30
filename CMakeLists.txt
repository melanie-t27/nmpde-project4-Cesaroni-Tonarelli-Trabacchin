cmake_minimum_required(VERSION 3.12.0)
project(project_nmpde LANGUAGES CXX C)

include(common/cmake-common.cmake)
include_directories(
        src
)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
#add_definitions (-D CHECK_ACTIVATION_TIMES)
if(CHECK_ACTIVATION_TIMES STREQUAL "ON")
    message(STATUS "Compiling with activation times support")
    add_definitions (-D CHECK_ACTIVATION_TIMES)
    # Add logic here based on the value of MY_VARIABLE
endif()

add_executable(nmpde src/main.cpp
        src/MonodomainSolver/FESolver.cpp )
deal_ii_setup_target(nmpde)


# Define the custom command with shell evaluation
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/mesh_generated
  COMMAND ${CMAKE_COMMAND} -E env sh -c "mkdir -p ../meshes && gmsh mesh_nmpde.geo -parse_and_exit -setnumber lc \$MESH_STEP"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
  COMMENT "Generating the mesh with gmsh"
  VERBATIM
)

# Define the custom target
add_custom_target(
  GenerateMesh
  DEPENDS ${CMAKE_BINARY_DIR}/mesh_generated
)