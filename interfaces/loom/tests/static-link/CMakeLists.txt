cmake_minimum_required(VERSION 3.24)
project(loom_static_consumer LANGUAGES C CXX ASM)
# This is a consumer of the existing native CMake owners, not another build wrapper.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
get_filename_component(RUNTIMES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_subdirectory("${RUNTIMES_ROOT}/yolo26-raspberry-pi-5/src/runtime" yolo)
add_executable(loom_static_link ../static_link.c)
set_target_properties(loom_static_link PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11)
target_link_libraries(loom_static_link PRIVATE arm_yolo26n_static)
enable_testing()
add_test(NAME yolo_static_runtime COMMAND loom_static_link)
add_test(NAME static_bootstrap_symbols COMMAND "${CMAKE_COMMAND}"
  -DNM=${CMAKE_NM} -DBINARY=$<TARGET_FILE:loom_static_link>
  -DYOLO_ARCHIVE=$<TARGET_FILE:arm_yolo26n_static>
  -P "${CMAKE_CURRENT_SOURCE_DIR}/check_symbols.cmake")
