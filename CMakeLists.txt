cmake_minimum_required(VERSION 3.20)
project(cognicore VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
    tree_sitter
    GIT_REPOSITORY https://github.com/tree-sitter/tree-sitter.git
    GIT_TAG        v0.25.3
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR _cognicore_no_upstream_targets
)
FetchContent_Declare(
    ts_cpp
    GIT_REPOSITORY https://github.com/tree-sitter/tree-sitter-cpp.git
    GIT_TAG        v0.23.4
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR _cognicore_no_upstream_targets
)
FetchContent_Declare(
    ts_python
    GIT_REPOSITORY https://github.com/tree-sitter/tree-sitter-python.git
    GIT_TAG        v0.23.6
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR _cognicore_no_upstream_targets
)
FetchContent_Declare(
    ts_javascript
    GIT_REPOSITORY https://github.com/tree-sitter/tree-sitter-javascript.git
    GIT_TAG        v0.23.1
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR _cognicore_no_upstream_targets
)

FetchContent_Declare(
    ts_typescript
    GIT_REPOSITORY https://github.com/tree-sitter/tree-sitter-typescript.git
    GIT_TAG v0.23.2
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR _cognicore_no_upstream_targets
)

FetchContent_MakeAvailable(tree_sitter ts_cpp ts_python ts_javascript ts_typescript)

add_library(ts_bundle STATIC
    ${tree_sitter_SOURCE_DIR}/lib/src/lib.c
    ${ts_cpp_SOURCE_DIR}/src/parser.c
    ${ts_cpp_SOURCE_DIR}/src/scanner.c
    ${ts_python_SOURCE_DIR}/src/parser.c
    ${ts_python_SOURCE_DIR}/src/scanner.c
    ${ts_javascript_SOURCE_DIR}/src/parser.c
    ${ts_javascript_SOURCE_DIR}/src/scanner.c
    ${ts_typescript_SOURCE_DIR}/typescript/src/parser.c
    ${ts_typescript_SOURCE_DIR}/typescript/src/scanner.c
)
target_include_directories(ts_bundle PUBLIC
    ${tree_sitter_SOURCE_DIR}/lib/include
    ${tree_sitter_SOURCE_DIR}/lib/src
)
set_source_files_properties(${ts_cpp_SOURCE_DIR}/src/parser.c ${ts_cpp_SOURCE_DIR}/src/scanner.c PROPERTIES INCLUDE_DIRECTORIES "${ts_cpp_SOURCE_DIR}/src")
set_source_files_properties(${ts_python_SOURCE_DIR}/src/parser.c ${ts_python_SOURCE_DIR}/src/scanner.c PROPERTIES INCLUDE_DIRECTORIES "${ts_python_SOURCE_DIR}/src")
set_source_files_properties(${ts_javascript_SOURCE_DIR}/src/parser.c ${ts_javascript_SOURCE_DIR}/src/scanner.c PROPERTIES INCLUDE_DIRECTORIES "${ts_javascript_SOURCE_DIR}/src")
set_source_files_properties(${ts_typescript_SOURCE_DIR}/typescript/src/parser.c ${ts_typescript_SOURCE_DIR}/typescript/src/scanner.c PROPERTIES INCLUDE_DIRECTORIES "${ts_typescript_SOURCE_DIR}/typescript/src")
set_target_properties(ts_bundle PROPERTIES C_STANDARD 11)

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
    src/coherence.cpp
    src/graph_mmap.cpp
    src/text_parser.cpp
    src/realizer.cpp
    src/analogical_realizer.cpp
    src/morphological_realizer.cpp
    src/assistant.cpp
    src/context_memory.cpp
    src/discourse_planner.cpp
    src/document_planner.cpp
    src/code_ast.cpp
    src/code_synthesizer.cpp
    src/episodic_memory.cpp
    src/language_rules.cpp
    src/lexicon.cpp
    src/lexicalized_realizer.cpp
    src/math_solver.cpp
    src/model_bundle.cpp
)

if(USE_ROCM)
    find_package(hip REQUIRED)
    enable_language(HIP)
    list(APPEND COGNICORE_SOURCES src/backends/hip_backend.hip)
endif()

add_library(cognicore_lib STATIC ${COGNICORE_SOURCES})
target_include_directories(cognicore_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cognicore_lib PUBLIC ts_bundle)
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

add_executable(test_marco7_morph tests/test_marco7_morph.cpp)
target_link_libraries(test_marco7_morph PRIVATE cognicore_lib)
target_compile_options(test_marco7_morph PRIVATE ${COGNICORE_CXX_FLAGS})
add_test(NAME test_marco7_morph COMMAND test_marco7_morph)

# Assertions in the test suite must remain active in Release builds.
foreach(test_target test_vsa test_sdm test_tsetlin test_hip_equiv test_pipeline test_marco7_morph)
    target_compile_options(${test_target} PRIVATE -UNDEBUG)
endforeach()

foreach(milestone 8_code_ast 9_code_synth)
    add_executable(test_marco${milestone} tests/test_marco${milestone}.cpp)
    target_link_libraries(test_marco${milestone} PRIVATE cognicore_lib)
    target_compile_options(test_marco${milestone} PRIVATE ${COGNICORE_CXX_FLAGS} -UNDEBUG)
    target_compile_definitions(test_marco${milestone} PRIVATE COGNICORE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME test_marco${milestone} COMMAND test_marco${milestone})
endforeach()

foreach(milestone 10_symbolic 11_assistant)
    add_executable(test_marco${milestone} tests/test_marco${milestone}.cpp)
    target_link_libraries(test_marco${milestone} PRIVATE cognicore_lib)
    target_compile_options(test_marco${milestone} PRIVATE ${COGNICORE_CXX_FLAGS} -UNDEBUG)
    add_test(NAME test_marco${milestone} COMMAND test_marco${milestone})
endforeach()

add_executable(test_marco12_document tests/test_marco12_document.cpp)
target_link_libraries(test_marco12_document PRIVATE cognicore_lib)
target_compile_options(test_marco12_document PRIVATE ${COGNICORE_CXX_FLAGS} -UNDEBUG)
add_test(NAME test_marco12_document COMMAND test_marco12_document)

add_executable(test_marco13_lexicon tests/test_marco13_lexicon.cpp)
target_link_libraries(test_marco13_lexicon PRIVATE cognicore_lib)
target_compile_options(test_marco13_lexicon PRIVATE ${COGNICORE_CXX_FLAGS} -UNDEBUG)
add_test(NAME test_marco13_lexicon COMMAND test_marco13_lexicon)

add_executable(test_marco14_lexicalized tests/test_marco14_lexicalized.cpp)
target_link_libraries(test_marco14_lexicalized PRIVATE cognicore_lib)
target_compile_options(test_marco14_lexicalized PRIVATE ${COGNICORE_CXX_FLAGS} -UNDEBUG)
add_test(NAME test_marco14_lexicalized COMMAND test_marco14_lexicalized)

add_executable(test_marco15_coherence tests/test_marco15_coherence.cpp)
target_link_libraries(test_marco15_coherence PRIVATE cognicore_lib)
target_compile_options(test_marco15_coherence PRIVATE ${COGNICORE_CXX_FLAGS} -UNDEBUG)
add_test(NAME test_marco15_coherence COMMAND test_marco15_coherence)

add_executable(test_improvements tests/test_improvements.cpp)
target_link_libraries(test_improvements PRIVATE cognicore_lib)
target_compile_options(test_improvements PRIVATE ${COGNICORE_CXX_FLAGS} -UNDEBUG)
add_test(NAME test_improvements COMMAND test_improvements)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME test_cli_phase2 COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli_phase2.py $<TARGET_FILE:cognicore>)
endif()
# Keep tests reproducible when several build configurations run concurrently.
get_property(cognicore_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${cognicore_tests} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=4")
