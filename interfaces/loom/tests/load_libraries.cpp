// Check generic and named bootstraps from the YOLO26n shared library.
#include "loom_runtime.h"
#include <dlfcn.h>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    auto get = reinterpret_cast<decltype(&lrt_get_api)>(dlsym(library, "lrt_get_api"));
    auto version = static_cast<const uint32_t*>(dlsym(library, "lrt_runtime_version"));
    lrt_api api{};
    lrt_context context{};
    if (!get || !version || *version != LRT_RUNTIME_VERSION ||
        get(LRT_RUNTIME_VERSION, &api) != LRT_OK ||
        api.get_runtime_version() != LRT_RUNTIME_VERSION ||
        api.create_context(&context) != LRT_OK) return 1;
    auto named_get = reinterpret_cast<decltype(&lrt_get_api)>(dlsym(library, "y26_lrt_get_api"));
    lrt_api named{};
    if (!named_get || named_get(LRT_RUNTIME_VERSION, &named) != LRT_OK ||
        named.create_context != api.create_context || named.infer != api.infer ||
        named.get_runtime_version != api.get_runtime_version) return 1;
    lrt_capabilities caps{};
    if (api.get_capabilities(context, &caps) != LRT_OK ||
        api.destroy_context(context) != LRT_OK) return 1;
    dlclose(library);
    return 0;
}
