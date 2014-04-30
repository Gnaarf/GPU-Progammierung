project(sheet03_task08_SSAA)
cmake_minimum_required(VERSION 2.8)

set(SRC_LIST SSAA.cpp)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../CMakeFindFunctions ${CMAKE_MODULE_PATH})
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories(${GLUT_INCLUDE_PATH})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_PATH})

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(
                        ${PROJECT_NAME}
                        ${GLUT_LIBRARIES}
                        ${OPENGL_LIBRARIES}
                        ${GLEW_LIBRARY}
)
