project(sheet02_task06_VertexBufferObjects)
cmake_minimum_required(VERSION 2.8)

set(SRC_LIST VBO_Fractal.cpp)
set(SRC_LIST Triangle.h ${SRC_LIST})
set(SRC_LIST Triangle.cpp ${SRC_LIST})
set(SRC_LIST Vector.h ${SRC_LIST})
set(SRC_LIST Vector.cpp ${SRC_LIST})


set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../CMakeFindFunctions ${CMAKE_MODULE_PATH})
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${GLEW_INCLUDE_PATH})
include_directories(${GLUT_INCLUDE_PATH})
include_directories(${OPENGL_INCLUDE_DIR})



add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(
                        ${PROJECT_NAME}
                        ${GLUT_LIBRARIES}
                        ${OPENGL_LIBRARIES}
                        ${GLEW_LIBRARY}
)
