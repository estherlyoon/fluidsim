cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(fluid-sim LANGUAGES CXX )#CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

list(APPEND CMAKE_MODULE_PATH "/usr/share/SFML/cmake/Modules")
find_package(SFML REQUIRED network audio graphics window system)

FILE(GLOB cmakes ${CMAKE_SOURCE_DIR}/cmake/*.cmake)
FOREACH(cmake ${cmakes})
	INCLUDE(${cmake})
ENDFOREACH(cmake)
 
set(APP_FILES "src/app.cpp"
              "src/app.hpp"
              "src/main.cpp"
              "src/common.cpp"
              "src/common.hpp"
              "src/fluidsim.cpp"
              "src/fluidsim.hpp"
              "src/cpu_solver.cpp"
              "src/cpu_solver.hpp"
              ) 

add_executable(fluid-sim ${APP_FILES})

target_link_libraries(fluid-sim sfml-graphics sfml-audio sfml-window sfml-system)

