
cmake_minimum_required(VERSION 2.8)
 
project(MIRotation)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(MIRotation MACOSX_BUNDLE MIRotation.cxx)
target_link_libraries(MIRotation ${ITK_LIBRARIES})
