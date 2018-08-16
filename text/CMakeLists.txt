cmake_minimum_required (VERSION 2.6)
project (Text)


find_package(OpenCV  REQUIRED)

include_directories("/usr/include/python2.7")
include_directories("/soft/opencv/include")

link_directories("/usr/lib64/python2.7/config")
link_directories("/usr/local/lib64")
#link_directories("/soft/opencv/build/lib")

add_executable(Text callpy.cpp)

target_link_libraries(Text  ${OpenCV_LIBS}  python2.7)

