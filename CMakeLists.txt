cmake_minimum_required(VERSION 3.20)
project(cognicore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_ROCM "Enable AMD HIP/ROCm GPU acceleration" OFF)

find_package(OpenMP)

set(COGNICORE_CXX_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -O3
    -march=native
)

set(COGNICORE_SOURCES
    src/backends/cpu_backend.cpp
    src/vsa.cpp
    src/sdm.cpp
    src/tsetlin.cpp
    src/causal_graph.cpp
    src/text_parser.cpp
    src/realizer.cpp
    src/model_bundle.cpp
)

if(USE_ROCM)
    find_package(hip REQUIRED)
    enable_language(HIP)
    list(APPEND COGNICORE_SOURCES src/backends/hip_backend.hip)
endif()

add_library(cognicore_lib STATIC ${COGNICORE_SOURCES})
target_include_directories(cognicore_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(cognicore_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${COGNICORE_CXX_FLAGS}>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cognicore_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_ROCM)
    target_compile_definitions(cognicore_lib PUBLIC COGNICORE_USE_ROCM=1)
    target_link_libraries(cognicore_lib PUBLIC hip::host)
else()
    target_compile_definitions(cognicore_lib PUBLIC COGNICORE_USE_ROCM=0)
endif()

add_executable(cognicore src/main.cpp)
target_link_libraries(cognicore PRIVATE cognicore_lib)
target_compile_options(cognicore PRIVATE ${COGNICORE_CXX_FLAGS})

enable_testing()

add_executable(test_vsa tests/test_vsa.cpp)
target_link_libraries(test_vsa PRIVATE cognicore_lib)
target_compile_options(test_vsa PRIVATE ${COGNICORE_CXX_FLAGS})
add_test(NAME test_vsa COMMAND test_vsa)

add_executable(test_sdm tests/test_sdm.cpp)
target_link_libraries(test_sdm PRIVATE cognicore_lib)
target_compile_options(test_sdm PRIVATE ${COGNICORE_CXX_FLAGS})
add_test(NAME test_sdm COMMAND test_sdm)

add_executable(test_tsetlin tests/test_tsetlin.cpp)
target_link_libraries(test_tsetlin PRIVATE cognicore_lib)
target_compile_options(test_tsetlin PRIVATE ${COGNICORE_CXX_FLAGS})
add_test(NAME test_tsetlin COMMAND test_tsetlin)

add_executable(test_hip_equiv tests/test_hip_equiv.cpp)
target_link_libraries(test_hip_equiv PRIVATE cognicore_lib)
target_compile_options(test_hip_equiv PRIVATE ${COGNICORE_CXX_FLAGS})
add_test(NAME test_hip_equiv COMMAND test_hip_equiv)

add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE cognicore_lib)
target_compile_options(test_pipeline PRIVATE ${COGNICORE_CXX_FLAGS})
target_compile_definitions(test_pipeline PRIVATE COGNICORE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
add_test(NAME test_pipeline COMMAND test_pipeline)
