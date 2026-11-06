/* Runtime-specific LOOM bootstrap for static and shared linking. */
#ifndef Y26_LRT_H
#define Y26_LRT_H
#include "loom_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
LRT_EXPORT lrt_status LRT_CALL y26_lrt_get_api(uint32_t requested_version, lrt_api* out_api);
#ifdef __cplusplus
}
#endif
#endif
