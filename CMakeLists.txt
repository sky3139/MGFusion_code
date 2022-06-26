cmake_minimum_required(VERSION 3.5)

project(project)
# packages
SET(CMAKE_BUILD_TYPE "Debug")

# SET(CMAKE_BUILD_TYPE "Release")

FIND_PACKAGE(Eigen3 REQUIRED)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories($ENV{CUDA_PATH}/lib/x64)

FIND_PACKAGE(OpenCV REQUIRED)
message(<<--OpenCV_VERSION:${OpenCV_VERSION}-->>)
FIND_PACKAGE(GLUT REQUIRED)
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++14)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-prec-div=true;--expt-relaxed-constexpr;-w)#;-G;-g$

file(GLOB CURRENT_SOURCES  src/*.*
src/cuda/*.*
)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

# add_library(tool src/tool.cc src/fr_parser.cpp
#  src/Timer.cpp 
# )
include_directories(${OpenCV_INCLUDE_DIRS}
./src
)
# find_package(Pangolin REQUIRED)
# include_directories(${Pangolin_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIR})
# find_package(FREEGLUT REQUIRED)
# include_directories(${FREEGLUT_INCLUDE_DIR})
## OpenGL

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
find_package(glfw3 REQUIRED)


file(GLOB TOOL_SRCS  src/tool/**)
cuda_add_library(tool  ${TOOL_SRCS})
target_link_libraries(tool ${OpenCV_LIBS})

CUDA_ADD_EXECUTABLE(mvgrid src/app/mvgrid.cu src/app/ratcast.cu ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries (mvgrid ${OpenCV_LIBS} tool ${GLUT_LIBRARY}   ${GLEW_LIBRARIES}
${OPENGL_LIBRARY})

CUDA_ADD_EXECUTABLE(raycast src/app/raycast.cu ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries (raycast ${OpenCV_LIBS} tool  )


CUDA_ADD_EXECUTABLE(mgraycast src/app/mgraycast.cu ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries (mgraycast ${OpenCV_LIBS} tool  )
# CUDA_ADD_EXECUTABLE(rnew src/app/main1.cu ${CURRENT_HEADERS} ${CURRENT_SOURCES})
# target_link_libraries (rnew ${OpenCV_LIBS} tool  )

# CUDA_ADD_EXECUTABLE(dmv src/app/dmv.cu ${CURRENT_HEADERS} ${CURRENT_SOURCES})
# target_link_libraries (dmv ${OpenCV_LIBS} tool lz4 )
# CUDA_ADD_EXECUTABLE(project src/app/tsdf_tum.cu ${CURRENT_HEADERS} ${CURRENT_SOURCES})
# target_link_libraries (project ${OpenCV_LIBS} tool lz4 )



