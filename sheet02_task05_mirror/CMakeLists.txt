project(sheet02_task05_mirror)
cmake_minimum_required(VERSION 2.8)

set(SRC_LIST mirror.cpp)
set(SRC_LIST lodepng.h ${SRC_LIST})
set(SRC_LIST lodepng.cpp ${SRC_LIST})

file(COPY mirrorTexture.png DESTINATION ${CMAKE_BINARY_DIR})
file(COPY mirrorTexture2.png DESTINATION ${CMAKE_BINARY_DIR})

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../CMakeFindFunctions ${CMAKE_MODULE_PATH})
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${GLUT_INCLUDE_PATH})
include_directories(${OPENGL_INCLUDE_DIR})

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(
                        ${PROJECT_NAME}
                        ${GLUT_LIBRARIES}
                        ${OPENGL_LIBRARIES}
)
