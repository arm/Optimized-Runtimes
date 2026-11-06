/* The named getter must resolve without the generic shared-library bootstrap. */
#include "y26_or.h"
#include <stdint.h>
#include <string.h>

static int check(or_status (OR_CALL *get_api)(uint32_t, or_api *), or_api *api) {
    unsigned char before[sizeof(*api)];
    memset(api, 0xa5, sizeof(*api));
    memcpy(before, api, sizeof(*api));
    if (get_api(OR_RUNTIME_VERSION + 1, api) != OR_INCOMPATIBLE ||
        memcmp(before, api, sizeof(*api)) != 0 ||
        get_api(OR_RUNTIME_VERSION, 0) != OR_INVALID_ARGUMENT ||
        get_api(OR_RUNTIME_VERSION, api) != OR_OK ||
        api->get_runtime_version() != OR_RUNTIME_VERSION) return 1;
    return 0;
}
int main(void) {
    or_api yolo = {0};
    or_context context = 0;
    or_capabilities caps = {0};
    if (check(y26_or_get_api, &yolo)) return 1;
    if (yolo.create_context(&context) != OR_OK) return 2;
    if (yolo.get_capabilities(context, &caps) != OR_OK ||
        !caps.runtime_id || !caps.runtime_id[0]) return 3;
    if (yolo.destroy_context(context) != OR_OK) return 4;
    return 0;
}
