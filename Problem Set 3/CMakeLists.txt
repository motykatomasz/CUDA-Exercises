cmake_minimum_required(VERSION 3.10)

project(pset3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(DEFAULT_SOURCES
        # Other files
        main.cpp
        utils/utils.hpp utils/timer.hpp
        utils/compare.hpp utils/compare.cpp
        utils/loadSaveImage.hpp utils/loadSaveImage.cpp
        reference/reference_calc.hpp reference/reference_calc.cpp
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
            cuda/cuda_utils.cu cuda/cuda_utils.hpp
            cuda/cuda_func.cu cuda/cuda_func.hpp
            )

else ()
    message("Could not find CUDA support. Disabling CUDA sources.")
endif ()

include_directories(${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${DEFAULT_SOURCES} ${CUDA_SOURCES})

target_link_libraries(pset3 ${OpenCV_LIBS})

set_property(TARGET pset3 PROPERTY CXX_STANDARD 14)

target_compile_options(pset3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -g -G >)
