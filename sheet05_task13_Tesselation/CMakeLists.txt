project(sheet05_task13_Tesselation)
cmake_minimum_required(VERSION 2.8)

set(SRC_LIST elephant.cpp)
set(SRC_LIST Tesselation.cpp ${SRC_LIST})

set(SHADER_FILES Tesselation.vert)
set(SHADER_FILES Tesselation.dom ${SHADER_FILES})
set(SHADER_FILES Tesselation.hull ${SHADER_FILES})
set(SHADER_FILES Tesselation.frag ${SHADER_FILES})
file(COPY ${SHADER_FILES} DESTINATION ${CMAKE_BINARY_DIR})

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../CMakeFindFunctions ${CMAKE_MODULE_PATH})
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories(${GLUT_INCLUDE_PATH})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_PATH})

add_executable(${PROJECT_NAME} ${SRC_LIST} ${SHADER_FILES})

target_link_libraries(
                        ${PROJECT_NAME}
                        ${GLUT_LIBRARIES}
                        ${OPENGL_LIBRARIES}
                        ${GLEW_LIBRARY}
)
