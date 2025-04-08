cmake_minimum_required(VERSION 3.10)
project(graphzy C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Add compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g -O2")

# Find raylib package
# Method 1: Using pkg-config (preferred)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(RAYLIB QUIET raylib)
endif()

# Method 2: Manual find
if(NOT RAYLIB_FOUND)
    # Try finding raylib in common Homebrew location on macOS
    find_path(RAYLIB_INCLUDE_DIRS raylib.h
        PATHS
        /opt/homebrew/include
        /usr/local/include
        /usr/include
    )
    
    find_library(RAYLIB_LIBRARIES raylib
        PATHS
        /opt/homebrew/lib
        /usr/local/lib
        /usr/lib
    )
    
    if(RAYLIB_INCLUDE_DIRS AND RAYLIB_LIBRARIES)
        set(RAYLIB_FOUND TRUE)
    endif()
endif()

# Check if raylib was found
if(NOT RAYLIB_FOUND)
    message(FATAL_ERROR "raylib not found. Please install raylib or set its location manually.")
endif()

# Add main executable
add_executable(graphzy main.c)

# Include directories
target_include_directories(graphzy PRIVATE ${RAYLIB_INCLUDE_DIRS})

# Link libraries
target_link_libraries(graphzy ${RAYLIB_LIBRARIES})

# Add macOS-specific frameworks if on Apple platform
if(APPLE)
    target_link_libraries(graphzy 
        "-framework CoreVideo" 
        "-framework IOKit" 
        "-framework Cocoa" 
        "-framework GLUT" 
        "-framework OpenGL"
    )
endif()

# Add math library (Unix-like systems)
if(UNIX)
    target_link_libraries(graphzy m)
endif()

# Install target (optional)
install(TARGETS graphzy DESTINATION bin)

# Configure make run command
add_custom_target(run
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/graphzy
    DEPENDS graphzy
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running graphzy..."
) 