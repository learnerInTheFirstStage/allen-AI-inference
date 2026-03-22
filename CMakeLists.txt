cmake_minimum_required(VERSION 3.15)

project(MyInference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Set Qt path from Homebrew
set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/qt" ${CMAKE_PREFIX_PATH})

find_package(Qt6 REQUIRED COMPONENTS Widgets)

include_directories(include)

add_executable(MyCli apps/main.cpp)

add_executable(MyGui apps/gui_main.cpp)

target_link_libraries(MyGui PRIVATE Qt6::Widgets)

if (MSVC)
    target_compile_options(MyCli PRIVATE /W4)
    target_compile_options(MyGui PRIVATE /W4)
else()
    target_compile_options(MyCli PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(MyGui PRIVATE -Wall -Wextra -Wpedantic)
endif()

if (APPLE)
    target_compile_options(MyGui PRIVATE -O3 -march=native)
endif()