# Build the model-independent YOLO26n runtime for its one supported native
# target. This project owns dependency acquisition, native compilation, ABI
# verification, and installation. Model conversion remains a separate operation
# and does not rebuild this target.
cmake_minimum_required(VERSION 3.24)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION" yolo26n_version LIMIT_COUNT 1)
if(NOT yolo26n_version MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "VERSION must contain one semantic version")
endif()
project(yolo26n_runtime VERSION "${yolo26n_version}" LANGUAGES C CXX ASM)

include(CTest)
include(FetchContent)
include(GNUInstallDirs)

option(YOLO26N_ENABLE_COVERAGE "Instrument the runtime for source coverage" OFF)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
    OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  message(FATAL_ERROR "The YOLO26n runtime builds on Linux AArch64")
endif()
if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU"
    OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    OR NOT CMAKE_ASM_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR
    "The YOLO26n runtime requires the GNU C, C++, and assembler toolchain")
endif()
if(YOLO26N_ENABLE_COVERAGE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(FATAL_ERROR "YOLO26N_ENABLE_COVERAGE requires CMAKE_BUILD_TYPE=Debug")
endif()
find_program(PATCH_EXECUTABLE patch REQUIRED)
set(FETCHCONTENT_QUIET OFF)

# Pinned native dependencies
# Fetch and patch each archive without configuring its top-level project.
# XNNPACK is added once below, after its build options and source dependencies
# have been set.
FetchContent_Declare(cpuinfo
  URL https://codeload.github.com/pytorch/cpuinfo/tar.gz/f9a03241f8c3d4ed0c9728f5d70bff873d43d4e0
  URL_HASH SHA256=90e3f1a6c3671a0f94259a1e35ad0524ec9d6a8e9275aeb1e3af502341b9eb0e
  SOURCE_DIR "${CMAKE_BINARY_DIR}/sources/cpuinfo"
  SOURCE_SUBDIR _populate_only)
# The pinned XNNPACK revision selects this same revision from FXdiv's canonical
# upstream repository.
FetchContent_Declare(fxdiv
  URL https://codeload.github.com/Maratyszcza/FXdiv/tar.gz/b408327ac2a15ec3e43352421954f5b1967701d1
  URL_HASH SHA256=9ccf554541666b5c089ad5dd465141d671c99971f36d72f313652f5c49ffce14
  SOURCE_DIR "${CMAKE_BINARY_DIR}/sources/fxdiv"
  SOURCE_SUBDIR _populate_only)
FetchContent_Declare(pthreadpool
  URL https://codeload.github.com/google/pthreadpool/tar.gz/a56dcd79c699366e7ac6466792c3025883ff7704
  URL_HASH SHA256=0663f88fbfc3e7e2c3e2522e18a460af94367aade7a5caf20b037b2eb7940173
  SOURCE_DIR "${CMAKE_BINARY_DIR}/sources/pthreadpool"
  SOURCE_SUBDIR _populate_only
  PATCH_COMMAND "${PATCH_EXECUTABLE}" --strip=3 --batch --forward
    --input=${CMAKE_CURRENT_SOURCE_DIR}/patches/pthreadpool-spin-wait.patch)
FetchContent_Declare(xnnpack
  URL https://codeload.github.com/google/XNNPACK/tar.gz/1adaa7c709d4839d29e1f219cb962b01c9e6a905
  URL_HASH SHA256=a101582c57d3d45f9fdb4d89b7aad7b9dbe4b488b429ccde483e199d564a9c41
  SOURCE_DIR "${CMAKE_BINARY_DIR}/sources/xnnpack"
  SOURCE_SUBDIR _populate_only
  PATCH_COMMAND "${PATCH_EXECUTABLE}" --strip=3 --batch --forward
    --input=${CMAKE_CURRENT_SOURCE_DIR}/patches/xnnpack-graph-api-pruning.patch
  COMMAND "${PATCH_EXECUTABLE}" --strip=3 --batch --forward
    --input=${CMAKE_CURRENT_SOURCE_DIR}/patches/xnnpack-yolo26-graph-extensions.patch
  COMMAND "${PATCH_EXECUTABLE}" --strip=3 --batch --forward
    --input=${CMAKE_CURRENT_SOURCE_DIR}/patches/xnnpack-operator-pruning.patch
  COMMAND "${PATCH_EXECUTABLE}" --strip=3 --batch --forward
    --input=${CMAKE_CURRENT_SOURCE_DIR}/patches/xnnpack-yolo26-operator-extensions.patch
  COMMAND "${PATCH_EXECUTABLE}" --strip=3 --batch --forward
    --input=${CMAKE_CURRENT_SOURCE_DIR}/patches/xnnpack-yolo26-kernels.patch
  COMMAND "${PATCH_EXECUTABLE}" --strip=3 --batch --forward
    --input=${CMAKE_CURRENT_SOURCE_DIR}/patches/xnnpack-rpi5-bti-diagnostic.patch
  COMMAND "${PATCH_EXECUTABLE}" --strip=3 --batch --forward
    --input=${CMAKE_CURRENT_SOURCE_DIR}/patches/xnnpack-rpi5-workspace-alignment.patch)
FetchContent_MakeAvailable(cpuinfo fxdiv pthreadpool xnnpack)

# Raspberry Pi 5 XNNPACK configuration
#
# Point XNNPACK at the already-fetched supporting sources, exclude kernels for
# CPU features outside the fixed target, then add XNNPACK exactly once.
set(CPUINFO_SOURCE_DIR "${cpuinfo_SOURCE_DIR}" CACHE PATH "" FORCE)
set(FXDIV_SOURCE_DIR "${fxdiv_SOURCE_DIR}" CACHE PATH "" FORCE)
set(PTHREADPOOL_SOURCE_DIR "${pthreadpool_SOURCE_DIR}" CACHE PATH "" FORCE)
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_ALL_MICROKERNELS OFF CACHE BOOL "" FORCE)
# The fixed Raspberry Pi 5 target uses AArch64 NEON FP16 kernels. Do not build
# dispatch families for CPU features outside that target.
set(XNNPACK_ENABLE_ARM_BF16 OFF CACHE BOOL "" FORCE)
set(XNNPACK_ENABLE_ARM_DOTPROD OFF CACHE BOOL "" FORCE)
set(XNNPACK_ENABLE_ARM_FP16_SCALAR OFF CACHE BOOL "" FORCE)
set(XNNPACK_ENABLE_ARM_I8MM OFF CACHE BOOL "" FORCE)
set(XNNPACK_ENABLE_ARM_SME OFF CACHE BOOL "" FORCE)
set(XNNPACK_ENABLE_ARM_SME2 OFF CACHE BOOL "" FORCE)
set(XNNPACK_ENABLE_KLEIDIAI OFF CACHE BOOL "" FORCE)
set(XNNPACK_LIBRARY_TYPE static CACHE STRING "" FORCE)
add_subdirectory("${xnnpack_SOURCE_DIR}" xnnpack EXCLUDE_FROM_ALL)

# These four project-owned kernels enter XNNPACK through the kernel patch but
# are deliberately absent from XNNPACK's upstream source lists. Register them
# with the production microkernel target explicitly because the fixed runtime
# disables XNNPACK_BUILD_ALL_MICROKERNELS. The assembly kernels require the
# same Armv8.2-A FP16 ISA as the runtime.
set(yolo26n_xnnpack_microkernels
  "${xnnpack_SOURCE_DIR}/src/f16-igemm/f16-igemm-1x16-yolo26-silu-asm-aarch64-neonfp16arith-ld64.S"
  "${xnnpack_SOURCE_DIR}/src/f16-igemm/f16-igemm-6x16-yolo26-silu-asm-aarch64-neonfp16arith-cortex-a75.S"
  "${xnnpack_SOURCE_DIR}/src/f16-vsilu/f16-vsilu-neonfp16arith-rr2-p2-nr1fma-u32.c"
  "${xnnpack_SOURCE_DIR}/src/f32-vsilu/f32-vsilu-neonfma-rr1-lut64-p2-nr2recps-u16.c")
target_sources(xnnpack-microkernels-prod PRIVATE ${yolo26n_xnnpack_microkernels})
set_source_files_properties(${yolo26n_xnnpack_microkernels}
  TARGET_DIRECTORY xnnpack-microkernels-prod
  PROPERTIES
  COMPILE_OPTIONS "-march=armv8.2-a+fp16")

# Runtime targets
#
# The shared-library export map keeps every C++ and dependency symbol private.
add_library(arm_yolo26n SHARED
  fixed_graph.cpp
  fixed_memory_plan.cpp
  fixed_schedule.cpp
  model_store.cpp
  arm_yolo26n.cpp
  preprocess.cpp
  projection_optimization.cpp)
set_target_properties(arm_yolo26n PROPERTIES
  OUTPUT_NAME "arm-yolo26n"
  VERSION "${PROJECT_VERSION}"
  SOVERSION 0)
target_compile_features(arm_yolo26n PRIVATE cxx_std_17)
target_compile_options(arm_yolo26n PRIVATE
  -march=armv8.2-a+fp16
  -mno-fix-cortex-a53-843419 -ffunction-sections -fdata-sections
  -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror)
target_include_directories(arm_yolo26n SYSTEM PRIVATE
  "${xnnpack_SOURCE_DIR}"
  "${xnnpack_SOURCE_DIR}/include"
  "${pthreadpool_SOURCE_DIR}/include")
target_link_libraries(arm_yolo26n PRIVATE XNNPACK pthreadpool m)
target_link_options(arm_yolo26n PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/native_runtime.map"
  "LINKER:--gc-sections"
  "LINKER:--sort-section=alignment"
  "LINKER:-Bsymbolic"
  "LINKER:-Map,${CMAKE_BINARY_DIR}/libarm-yolo26n.map"
  "LINKER:--cref")
if(YOLO26N_ENABLE_COVERAGE)
  # Debug supplies unoptimized code and symbols; this option adds only the GNU
  # coverage instrumentation. Coverage builds are diagnostic and are never
  # used for performance measurement or release.
  target_compile_options(arm_yolo26n PRIVATE --coverage)
  target_link_options(arm_yolo26n PRIVATE --coverage)
endif()

# Native verification
if(BUILD_TESTING)
  # Inspect the built library's dynamic dependencies and exported symbols
  # without loading it. The C API test below exercises function behavior.
  add_test(NAME yolo26n_runtime_binary
    COMMAND "${CMAKE_COMMAND}"
      -DNATIVE_RUNTIME=$<TARGET_FILE:arm_yolo26n>
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_runtime_binary.cmake")

  # Compile the repository's only C API behavior test against the public
  # header. It covers invalid calls that the higher-level Python API prevents
  # from reaching the native boundary.
  add_executable(yolo26n_runtime_api_test tests/runtime_api_test.c)
  target_compile_features(yolo26n_runtime_api_test PRIVATE c_std_11)
  target_compile_options(yolo26n_runtime_api_test PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror)
  target_include_directories(yolo26n_runtime_api_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(yolo26n_runtime_api_test PRIVATE arm_yolo26n)

  # The behavior test returns 77 when no converted model was supplied. CTest
  # reports that result as skipped, allowing an ordinary native build to run
  # every available check without making CMake own model conversion. The
  # end-to-end and coverage workflows supply the model and exercise the test.
  add_test(NAME yolo26n_runtime_api_behavior COMMAND yolo26n_runtime_api_test)
  set_tests_properties(yolo26n_runtime_api_behavior PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Installation
install(TARGETS arm_yolo26n
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  COMPONENT runtime
  NAMELINK_COMPONENT development)
install(FILES arm_yolo26n.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT development)
install(DIRECTORY licenses/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/arm-yolo26n/licenses"
  COMPONENT runtime)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../../LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/arm-yolo26n"
  COMPONENT runtime)
