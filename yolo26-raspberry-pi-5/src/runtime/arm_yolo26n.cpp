#include "arm_yolo26n.h"

#include <pthreadpool.h>
#include <xnnpack.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "buffer_validation.h"
#include "fixed_graph.h"
#include "fixed_schedule.h"
#include "model_store.h"

namespace {

// Owns the complete lifetime of one runtime instance. Members are declared in
// dependency order; explicit teardown releases graph runtimes before their
// shared workspace, threadpool, scratch buffers, and model storage disappear.
struct NativeRuntime {
  NativeRuntime() = default;
  NativeRuntime(const NativeRuntime&) = delete;
  NativeRuntime& operator=(const NativeRuntime&) = delete;
  ~NativeRuntime() {
    for (xnn_runtime_t runtime : runtimes) {
      if (runtime) xnn_delete_runtime(runtime);
    }
    if (workspace) xnn_release_workspace(workspace);
    if (threadpool) pthreadpool_destroy(threadpool);
    std::free(arena);
    std::free(project);
  }

  ModelStore model;
  ScheduleState schedule;
  std::array<xnn_runtime_t, kGraphCount> runtimes{};
  xnn_workspace_t workspace = nullptr;
  pthreadpool_t threadpool = nullptr;
  uint8_t* arena = nullptr;
  uint16_t* project = nullptr;
  std::atomic<uint64_t> calls{0};
};

bool Allocate(void** output, size_t bytes) {
  return bytes != 0 && posix_memalign(output, yolo26::kTensorAlignment, bytes) == 0;
}

bool Initialize(NativeRuntime* native, const void* model_data, size_t model_size) {
  // Validate model-derived dimensions before creating any native execution
  // resources. The destructor handles every partially initialized state.
  if (!native->model.Load(model_data, model_size) ||
      !InitializeSchedule(native->model.resolution(), native->model.class_count(),
                          &native->schedule) ||
      xnn_initialize(nullptr) != xnn_status_success ||
      xnn_create_workspace(&native->workspace) != xnn_status_success) {
    return false;
  }
  native->threadpool = pthreadpool_create(yolo26::kThreadCount);
  if (!native->threadpool ||
      pthreadpool_get_threads_count(native->threadpool) != yolo26::kThreadCount ||
      !Allocate(reinterpret_cast<void**>(&native->arena), native->schedule.arena_bytes) ||
      !Allocate(reinterpret_cast<void**>(&native->project), native->schedule.project_bytes)) {
    return false;
  }
  const GraphContext graphs{
      &native->model,
      native->workspace,
      native->threadpool,
      &native->runtimes,
  };
  return BuildFixedGraphs(graphs) && ConfigureFixedProjection(native->runtimes);
}

}  // namespace

extern "C" y26_native_runtime* y26_native_create(const void* model_data, size_t model_size) {
  if (!model_data) return nullptr;
  auto runtime = std::make_unique<NativeRuntime>();
  if (!Initialize(runtime.get(), model_data, model_size)) return nullptr;
  return reinterpret_cast<y26_native_runtime*>(runtime.release());
}

extern "C" y26_native_status y26_native_execute(y26_native_runtime* runtime, const float* input,
                                                size_t input_bytes, float* output,
                                                size_t output_bytes) {
  auto* native = reinterpret_cast<NativeRuntime*>(runtime);
  if (!native || !input || !output || input_bytes != native->schedule.input_bytes ||
      output_bytes != native->schedule.output_bytes ||
      yolo26::BuffersOverlap(input, input_bytes, output, output_bytes)) {
    return Y26_NATIVE_INVALID_ARGUMENT;
  }
  if (!RunFixedSchedule(native->schedule, native->runtimes, native->threadpool, native->arena,
                        native->project, input, output)) {
    return Y26_NATIVE_EXECUTION_FAILED;
  }
  if (!OutputIsFinite(native->schedule, output)) return Y26_NATIVE_NONFINITE_OUTPUT;
  ++native->calls;
  return Y26_NATIVE_OK;
}

extern "C" void y26_native_destroy(y26_native_runtime* runtime) {
  delete reinterpret_cast<NativeRuntime*>(runtime);
}

extern "C" uint64_t y26_native_calls(const y26_native_runtime* runtime) {
  const auto* native = reinterpret_cast<const NativeRuntime*>(runtime);
  return native ? native->calls.load() : 0;
}

extern "C" uint32_t y26_native_graphs(const y26_native_runtime* runtime) {
  return runtime ? kGraphCount : 0;
}

extern "C" uint32_t y26_native_resolution(const y26_native_runtime* runtime) {
  const auto* native = reinterpret_cast<const NativeRuntime*>(runtime);
  return native ? native->model.resolution() : 0;
}

extern "C" uint32_t y26_native_class_count(const y26_native_runtime* runtime) {
  const auto* native = reinterpret_cast<const NativeRuntime*>(runtime);
  return native ? native->model.class_count() : 0;
}

extern "C" size_t y26_native_input_bytes(const y26_native_runtime* runtime) {
  const auto* native = reinterpret_cast<const NativeRuntime*>(runtime);
  return native ? native->schedule.input_bytes : 0;
}

extern "C" size_t y26_native_output_bytes(const y26_native_runtime* runtime) {
  const auto* native = reinterpret_cast<const NativeRuntime*>(runtime);
  return native ? native->schedule.output_bytes : 0;
}

extern "C" size_t y26_native_arena_bytes(const y26_native_runtime* runtime) {
  const auto* native = reinterpret_cast<const NativeRuntime*>(runtime);
  return native ? native->schedule.arena_bytes : 0;
}
