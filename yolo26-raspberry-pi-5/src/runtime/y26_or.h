/* Runtime-specific OR bootstrap for static and shared linking. */
#ifndef Y26_OR_H
#define Y26_OR_H
#include "or_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
OR_EXPORT or_status OR_CALL y26_or_get_api(uint32_t requested_version, or_api* out_api);
#ifdef __cplusplus
}
#endif
#endif
