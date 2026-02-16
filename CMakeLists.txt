cmake_minimum_required(VERSION 3.10)
project(OpenCV_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

file(GLOB SOURCES "src/*.cpp")

add_executable(OpenCV_Project ${SOURCES})

target_include_directories(OpenCV_Project PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(OpenCV_Project ${OpenCV_LIBS})
