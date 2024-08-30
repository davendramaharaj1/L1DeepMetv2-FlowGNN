cmake_minimum_required(VERSION 3.14)
project(GraphMetNetwork)

# Find the pybind11 package
find_package(pybind11 REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the source files
set(SOURCES
    GraphMetNetwork.cpp
    test_main.cpp  # This file contains the Pybind11 bindings
)

# Add the Pybind11 module
pybind11_add_module(graphmetnetwork ${SOURCES})

# Specify include directories (if necessary)
# target_include_directories(graphmetnetwork PRIVATE ${CMAKE_SOURCE_DIR}/include)
