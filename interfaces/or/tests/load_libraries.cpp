// Check generic and named bootstraps from the YOLO26n shared library.
#include "or_runtime.h"
#include <dlfcn.h>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    auto get = reinterpret_cast<decltype(&or_get_api)>(dlsym(library, "or_get_api"));
    auto version = static_cast<const uint32_t*>(dlsym(library, "or_runtime_version"));
    or_api api{};
    or_context context{};
    if (!get || !version || *version != OR_RUNTIME_VERSION ||
        get(OR_RUNTIME_VERSION, &api) != OR_OK ||
        api.get_runtime_version() != OR_RUNTIME_VERSION ||
        api.create_context(&context) != OR_OK) return 1;
    auto named_get = reinterpret_cast<decltype(&or_get_api)>(dlsym(library, "y26_or_get_api"));
    or_api named{};
    if (!named_get || named_get(OR_RUNTIME_VERSION, &named) != OR_OK ||
        named.create_context != api.create_context || named.infer != api.infer ||
        named.get_runtime_version != api.get_runtime_version) return 1;
    or_capabilities caps{};
    if (api.get_capabilities(context, &caps) != OR_OK ||
        api.destroy_context(context) != OR_OK) return 1;
    dlclose(library);
    return 0;
}
