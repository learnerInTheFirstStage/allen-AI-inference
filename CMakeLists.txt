cmake_minimum_required(VERSION 3.15)
project(MyInference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Qt Configuration ---
# Only try to find Qt if we aren't in a headless environment (like basic WSL)
if(APPLE)
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/qt" ${CMAKE_PREFIX_PATH})
endif()

# We make Qt optional so your CLI tool can still compile on WSL2 without a GUI setup
find_package(Qt6 COMPONENTS Widgets QUIET)

include_directories(include)

# --- Targets ---
add_executable(MyCli apps/main.cpp)

if(Qt6_FOUND)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)
    add_executable(MyGui apps/gui_main.cpp)
    target_link_libraries(MyGui PRIVATE Qt6::Widgets)
else()
    message(STATUS "Qt6 not found. Skipping MyGui target.")
endif()

# --- Compiler Flags ---
if (MSVC)
    # Windows
    target_compile_options(MyCli PRIVATE /W4 /O2)
    target_compile_options(MyGui PRIVATE /W4 /O2)
else()
    # WSL2: GCC || Mac: Clang
    target_compile_options(MyCli PRIVATE -Wall -O3)
    if (APPLE)
        target_compile_options(MyCli PRIVATE -march=native)
    endif()
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        target_compile_options(MyCli PRIVATE -march=haswell)
    endif()
endif()