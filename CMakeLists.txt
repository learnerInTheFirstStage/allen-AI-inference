cmake_minimum_required(VERSION 3.15)

project(MyInference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(MyApp apps/main.cpp)

if (MSVC)
    target_compile_options(MyApp PRIVATE /W4)
else()
    target_compile_options(MyApp PRIVATE -Wall -Wextra -Wpedantic)
endif()