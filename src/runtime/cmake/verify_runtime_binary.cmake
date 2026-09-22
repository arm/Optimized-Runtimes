# Verify the deployment boundary of a built native runtime binary. readelf
# inspects the dynamic section without loading the library and confirms that it
# names only the expected system dependencies. nm confirms that the library
# exports exactly the public C ABI. CTest invokes this script for the library
# produced by the CMake target.

if(NOT DEFINED NATIVE_RUNTIME OR NOT EXISTS "${NATIVE_RUNTIME}")
  message(FATAL_ERROR "Native runtime does not exist: ${NATIVE_RUNTIME}")
endif()

find_program(READELF_EXECUTABLE readelf REQUIRED)
find_program(NM_EXECUTABLE nm REQUIRED)

# Dynamic-library boundary
execute_process(
  COMMAND "${READELF_EXECUTABLE}" --dynamic --wide "${NATIVE_RUNTIME}"
  OUTPUT_VARIABLE dynamic_section
  COMMAND_ERROR_IS_FATAL ANY)
set(expected_libraries
  libc.so.6
  libgcc_s.so.1
  libm.so.6
  libstdc++.so.6)
set(actual_libraries)
string(REPLACE "\n" ";" dynamic_section_lines "${dynamic_section}")
foreach(line IN LISTS dynamic_section_lines)
  if(line MATCHES "\\(NEEDED\\).*\\[([^]]+)\\]")
    list(APPEND actual_libraries "${CMAKE_MATCH_1}")
  endif()
endforeach()
list(SORT actual_libraries)
if(NOT actual_libraries STREQUAL expected_libraries)
  message(FATAL_ERROR
    "Unexpected native runtime dependencies. "
    "Expected: ${expected_libraries}. Actual: ${actual_libraries}")
endif()

# Public-symbol boundary
execute_process(
  COMMAND "${NM_EXECUTABLE}" -D --defined-only "${NATIVE_RUNTIME}"
  OUTPUT_VARIABLE nm_output
  COMMAND_ERROR_IS_FATAL ANY)
string(REGEX MATCHALL "[0-9A-Fa-f]+ [A-Za-z] ([A-Za-z0-9_]+)" symbol_rows "${nm_output}")
set(actual_exports)
foreach(row IN LISTS symbol_rows)
  string(REGEX REPLACE ".* " "" symbol "${row}")
  list(APPEND actual_exports "${symbol}")
endforeach()
list(SORT actual_exports)
set(expected_exports
  y26_native_arena_bytes
  y26_native_calls
  y26_native_class_count
  y26_native_create
  y26_native_destroy
  y26_native_execute
  y26_native_graphs
  y26_native_input_bytes
  y26_native_output_bytes
  y26_native_preprocess_image_no_resize
  y26_native_resolution)
if(NOT actual_exports STREQUAL expected_exports)
  message(FATAL_ERROR "Unexpected native runtime exports: ${actual_exports}")
endif()
