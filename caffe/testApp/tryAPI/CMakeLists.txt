cmake_minimum_required(VERSION 2.8.3)
project(classify)
#设置caffe的根目录，需改成你的路径
set(Caffe_DIR /home/lyoid/caffe/cbuild)
#设置OpenCV的根目录，需改成你的路径
set(OpenCV_DIR /home/lyoid/下载/opencv-3.4.0/build)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include
        ${OpenCV_INCLUDE_DIRS}
        ${Caffe_INCLUDE_DIRS}
        )

set(CPP_SOURCES src/classify.cpp)

add_executable(classify ${CPP_SOURCES})
target_link_libraries(classify
        ${OpenCV_LIBS}
        ${Caffe_LIBRARIES}
        )
