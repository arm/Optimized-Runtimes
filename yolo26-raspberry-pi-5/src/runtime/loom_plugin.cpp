// Generic plugin symbols belong only to the shared library.
#include "y26_lrt.h"
extern "C" {
extern const uint32_t lrt_runtime_version = LRT_RUNTIME_VERSION;
lrt_status LRT_CALL lrt_get_api(uint32_t requested_version, lrt_api* out_api) {
  return y26_lrt_get_api(requested_version, out_api);
}
}
