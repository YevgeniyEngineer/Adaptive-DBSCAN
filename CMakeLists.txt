cmake_minimum_required(VERSION 3.18)

project(adaptive_dbscan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nanoflann REQUIRED)

add_library(${PROJECT_NAME}
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/adaptive_dbscan_pointcloud.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/adaptive_dbscan_clustering.hpp
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(${PROJECT_NAME}
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
    INTERFACE
    nanoflann::nanoflann
)

add_library(clustering::adaptive_dbscan
    ALIAS
    ${PROJECT_NAME}
)

add_executable(adaptive_dbscan_example main.cpp)

target_link_libraries(adaptive_dbscan_example
    clustering::adaptive_dbscan
)