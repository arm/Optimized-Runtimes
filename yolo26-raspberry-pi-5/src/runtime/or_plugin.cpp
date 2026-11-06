// Generic plugin symbols belong only to the shared library.
#include "y26_or.h"
extern "C" {
extern const uint32_t or_runtime_version = OR_RUNTIME_VERSION;
or_status OR_CALL or_get_api(uint32_t requested_version, or_api* out_api) {
  return y26_or_get_api(requested_version, out_api);
}
}
