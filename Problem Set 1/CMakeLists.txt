cmake_minimum_required(VERSION 3.10)

project(pset1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")

set(DEFAULT_SOURCES
        # Other files
        main.cpp
        utils/utils.hpp utils/utils.cpp
        utils/timer.hpp utils/timer.cpp
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

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


add_executable(${PROJECT_NAME} ${DEFAULT_SOURCES} ${CUDA_SOURCES})

target_link_libraries(pset1 ${OpenCV_LIBS})
target_compile_options(pset1 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -g
                       -G
                       >)
