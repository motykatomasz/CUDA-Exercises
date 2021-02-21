cmake_minimum_required(VERSION 3.10)

project(pset2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")

set(DEFAULT_SOURCES
        # Other files
        main.cpp
        utils/utils.hpp utils/timer.hpp
        opencv/opencv_func.hpp opencv/opencv_func.cpp
)

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    message("CUDA is supported. Enabling CUDA sources.")
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CUDA_SOURCES
            # ADD YOUR OWN CUDA SOURCES HERE
            cuda/cuda_func.hpp cuda/cuda_func.cu
            )

else ()
    message("Could not find CUDA support. Disabling CUDA sources.")
endif ()

include_directories(${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${DEFAULT_SOURCES} ${CUDA_SOURCES})

target_link_libraries(pset2 ${OpenCV_LIBS})
target_compile_options(pset2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -g
                       -G
                       >)
