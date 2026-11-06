/* The named getter must resolve without the generic shared-library bootstrap. */
#include "y26_lrt.h"
#include <stdint.h>
#include <string.h>

static int check(lrt_status (LRT_CALL *get_api)(uint32_t, lrt_api *), lrt_api *api) {
    unsigned char before[sizeof(*api)];
    memset(api, 0xa5, sizeof(*api));
    memcpy(before, api, sizeof(*api));
    if (get_api(LRT_RUNTIME_VERSION + 1, api) != LRT_INCOMPATIBLE ||
        memcmp(before, api, sizeof(*api)) != 0 ||
        get_api(LRT_RUNTIME_VERSION, 0) != LRT_INVALID_ARGUMENT ||
        get_api(LRT_RUNTIME_VERSION, api) != LRT_OK ||
        api->get_runtime_version() != LRT_RUNTIME_VERSION) return 1;
    return 0;
}
int main(void) {
    lrt_api yolo = {0};
    lrt_context context = 0;
    lrt_capabilities caps = {0};
    if (check(y26_lrt_get_api, &yolo)) return 1;
    if (yolo.create_context(&context) != LRT_OK) return 2;
    if (yolo.get_capabilities(context, &caps) != LRT_OK ||
        !caps.runtime_id || !caps.runtime_id[0]) return 3;
    if (yolo.destroy_context(context) != LRT_OK) return 4;
    return 0;
}
