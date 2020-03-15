cmake_minimum_required(VERSION 3.1)
project(Sens)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")  # enable assert
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context -Wsign-compare")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")
endif()

option(BUILD_PERFORMANCE_TEST "More subdivisition for performance test" OFF)
option(BUILD_OPENMP "Enable support for OpenMP" OFF)
option(BUILD_LOG "Enable verbose log" OFF)

include_directories(.)
include_directories(/orion/u/jingweih/3rd/opencv/include)
include_directories(/orion/u/jingweih/3rd/glm)
link_directories(/orion/u/jingweih/3rd/opencv/lib)
set(
    sens_SRC
    parser.cpp
)

add_library(
    Sens SHARED
    ${sens_SRC}
)

target_link_libraries(
    Sens
    opencv_core
    opencv_highgui
    opencv_imgproc
)

set(
    rasterizer_SRC
    rasterizer.cpp
)

add_library(
    Rasterizer SHARED
    ${rasterizer_SRC}
)

target_link_libraries(
    Rasterizer
)

set(
    paint_SRC
    paint.cpp
)

add_library(
    Painter SHARED
    ${paint_SRC}
)

target_link_libraries(
    Painter
)
