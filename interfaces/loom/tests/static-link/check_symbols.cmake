execute_process(COMMAND "${NM}" -g --defined-only "${BINARY}"
  OUTPUT_VARIABLE symbols COMMAND_ERROR_IS_FATAL ANY)
if(symbols MATCHES "[ \t](lrt_get_api|lrt_runtime_version)([\r\n]|$)")
  message(FATAL_ERROR "Generic plugin bootstrap leaked into static consumer")
endif()
# LTO may internalize a named getter in the final executable. Inspect the input
# archives when supplied so optimization is not mistaken for a missing API.
set(name y26_lrt_get_api)
set(archive "${YOLO_ARCHIVE}")
set(exports "${symbols}")
if(archive)
  execute_process(COMMAND "${NM}" -g --defined-only "${archive}"
    OUTPUT_VARIABLE exports COMMAND_ERROR_IS_FATAL ANY)
  if(exports MATCHES "[ \t](lrt_get_api|lrt_runtime_version)([\r\n]|$)")
    message(FATAL_ERROR "Generic plugin bootstrap leaked into YOLO26n archive")
  endif()
endif()
if(NOT exports MATCHES "[ \t]${name}([\r\n]|$)")
  message(FATAL_ERROR "Missing runtime-specific getter: ${name}")
endif()
