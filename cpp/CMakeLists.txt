cmake_minimum_required(VERSION 3.14)
project(concurrent_backtesting VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find threads library
find_package(Threads REQUIRED)

# Main executable
add_executable(model_run 
    src/main.cpp
)

target_include_directories(model_run PRIVATE src)
target_link_libraries(model_run PRIVATE Threads::Threads)

# Installation
install(TARGETS model_run DESTINATION bin)

# Add a custom target to export models
add_custom_target(export_models
    COMMAND python3 ${CMAKE_SOURCE_DIR}/../training/export_models.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../training
    COMMENT "Exporting Python models to JSON format"
)
