// Synchronous host-memory implementation of the shared OR ABI.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "arm_yolo26n.h"
#include "model_store.h"
#include "y26_or.h"

namespace {
struct Context {
  size_t children = 0;
};
struct Model {
  or_context context;
  std::vector<uint8_t> data;
  std::string id = "yolo26n-xnnpack";
  size_t sessions = 0;
  uint64_t dims[2][4]{};
  int64_t sdims[2][4]{};
  uint64_t strides[2][4]{};
  uint64_t bytes[2]{};
  unsigned ranks[2] = {4, 3};
  or_memory_option options[2]{};
  or_tensor_requirement reqs[2]{};
  const or_tensor_requirement *req_in[1]{}, *req_out[1]{};
  or_tensor_info infos[2]{};
  const or_tensor_info *info_in[1]{}, *info_out[1]{};
  Model(or_context c, const std::vector<uint8_t>& d) : context(c), data(d) {
    ModelStore store;
    if (!store.Load(d.data(), d.size())) throw std::runtime_error("Invalid model.bin");
    const uint64_t r = store.resolution();
    const uint64_t positions = (r / 8) * (r / 8) + (r / 16) * (r / 16) + (r / 32) * (r / 32);
    dims[0][0] = 1;
    dims[0][1] = 3;
    dims[0][2] = r;
    dims[0][3] = r;
    dims[1][0] = 1;
    dims[1][1] = 4 + store.class_count();
    dims[1][2] = positions;
    const or_memory_configuration cfg = {OR_OWNER_RUNTIME, OR_ACCESSIBLE_HOST,
                                          OR_REFERENCE_POINTER, OR_TRANSFER_DIRECT};
    for (unsigned k = 0; k < 2; ++k) {
      uint64_t extent = sizeof(float);
      for (unsigned i = ranks[k]; i-- > 0;) {
        sdims[k][i] = static_cast<int64_t>(dims[k][i]);
        strides[k][i] = extent;
        extent *= dims[k][i];
      }
      bytes[k] = extent;
      options[k] = {cfg,
                    k == 0 ? OR_HANDOFF_NONE : OR_HANDOFF_BORROWED,
                    extent,
                    yolo26::kTensorAlignment,
                    OR_POINTER_RUNTIME_ALLOCATION,
                    nullptr};
      reqs[k] = {{ranks[k], dims[k]}, 0, extent,     yolo26::kTensorAlignment,
                 strides[k],          1, &options[k]};
      infos[k] = {k == 0 ? "images" : "predictions",
                  OR_FLOAT32,
                  ranks[k],
                  sdims[k],
                  dims[k],
                  dims[k],
                  k == 0 ? "tensor/nchw/v1" : "tensor/dense/v1",
                  {0, -1, nullptr, nullptr}};
    }
    req_in[0] = &reqs[0];
    req_out[0] = &reqs[1];
    info_in[0] = &infos[0];
    info_out[0] = &infos[1];
  }
};
struct Buffer {
  or_context context;
  void* data;
  uint64_t bytes;
  or_access access;
  unsigned bound = 0;
  ~Buffer() { std::free(data); }
};
struct Bound {
  or_buffer buffer = nullptr;
  uint64_t offset = 0, bytes = 0;
};
struct Session {
  or_model model;
  std::unique_ptr<y26_native_runtime, decltype(&y26_native_destroy)> native;
  std::array<Bound, 2> bound{};
  bool validated = false, invalid = false, running = false;
  size_t completions = 0;
  uint64_t generation = 0;
  std::array<or_output, 1> outputs{};
  const or_output* output_ptrs[1]{};
  explicit Session(or_model m);
};
struct Completion {
  or_session session;
  uint64_t generation;
  or_status result;
};

Context* object(or_context handle) { return reinterpret_cast<Context*>(handle); }
Model* object(or_model handle) { return reinterpret_cast<Model*>(handle); }
Buffer* object(or_buffer handle) { return reinterpret_cast<Buffer*>(handle); }
Session* object(or_session handle) { return reinterpret_cast<Session*>(handle); }
Completion* object(or_completion handle) { return reinterpret_cast<Completion*>(handle); }
Session::Session(or_model m)
    : model(m),
      native(y26_native_create(object(m)->data.data(), object(m)->data.size()),
             y26_native_destroy) {
  if (!native) throw std::runtime_error("YOLO session creation failed");
}
std::unordered_set<or_context> contexts;
std::unordered_set<or_model> models;
std::unordered_set<or_buffer> buffers;
std::unordered_set<or_session> sessions;
std::unordered_set<or_completion> completions;
thread_local char error[256]{};
std::recursive_mutex registry_mutex;
constexpr or_memory_configuration config = {OR_OWNER_RUNTIME, OR_ACCESSIBLE_HOST,
                                             OR_REFERENCE_POINTER, OR_TRANSFER_DIRECT};
or_status status(or_status x) noexcept {
  if (x != OR_OK) std::snprintf(error, sizeof(error), "YOLO OR status %d", x);
  return x;
}

template <class F>
or_status protect(F f) noexcept {
  try {
    return status(f());
  } catch (const std::bad_alloc&) {
    return status(OR_OUT_OF_MEMORY);
  } catch (const std::exception& e) {
    std::snprintf(error, sizeof(error), "%s", e.what());
    return OR_EXECUTION_FAILED;
  } catch (...) {
    return status(OR_EXECUTION_FAILED);
  }
}

bool same(const or_memory_configuration& c) {
  return c.owner == config.owner && c.accessibility == config.accessibility &&
         c.reference == config.reference && c.transfer == config.transfer;
}

bool shape_ok(or_model m, or_shape s, unsigned k, const uint64_t* st = nullptr) {
  if (s.rank != object(m)->ranks[k] || !s.dimensions) return false;
  for (unsigned i = 0; i < s.rank; ++i)
    if (s.dimensions[i] != object(m)->dims[k][i] || (st && st[i] != object(m)->strides[k][i]))
      return false;
  return true;
}

int slot(or_io io, uint32_t i) {
  return io == OR_INPUT ? (i == 0 ? 0 : -1) : io == OR_OUTPUT ? (i < 1 ? int(i + 1) : -1) : -1;
}

void unbind(or_session s, unsigned k) {
  if (object(s)->bound[k].buffer) --object(object(s)->bound[k].buffer)->bound;
  object(s)->bound[k] = {};
  object(s)->validated = false;
  ++object(s)->generation;
}

uint32_t version() { return OR_RUNTIME_VERSION; }

or_status create_context(or_context* out) {
  return protect([&]() -> or_status {
    if (!out) return OR_INVALID_ARGUMENT;
    auto p = std::make_unique<Context>();
    contexts.insert(reinterpret_cast<or_context>(p.get()));
    *out = reinterpret_cast<or_context>(p.release());
    return OR_OK;
  });
}

or_status capabilities(or_context c, or_capabilities* out) {
  if (!contexts.count(c) || !out) return status(OR_INVALID_ARGUMENT);
  *out = {OR_CAP_SYNC_INFERENCE, 4, "yolo26n-native-cpu", "XNNPACK-fixed-plan"};
  return OR_OK;
}

or_status load(or_context c, const or_model_source* src, or_model* out) {
  return protect([&]() -> or_status {
    if (!contexts.count(c) || !src || !out) return OR_INVALID_ARGUMENT;
    std::vector<uint8_t> data;
    if (src->kind == OR_MODEL_SOURCE_MEMORY) {
      if (!src->source.memory.data || !src->source.memory.byte_length ||
          src->source.memory.byte_length > std::numeric_limits<size_t>::max())
        return OR_INVALID_ARGUMENT;
      const auto* p = static_cast<const uint8_t*>(src->source.memory.data);
      data.assign(p, p + src->source.memory.byte_length);
    } else if (src->kind == OR_MODEL_SOURCE_PATH) {
      if (!src->source.path || !*src->source.path) return OR_INVALID_ARGUMENT;
      std::ifstream f(src->source.path, std::ios::binary | std::ios::ate);
      if (!f) return OR_IO_ERROR;
      if (f.tellg() <= 0) return OR_INCOMPATIBLE;
      data.resize(static_cast<size_t>(f.tellg()));
      f.seekg(0);
      f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
      if (!f) return OR_IO_ERROR;
    } else
      return OR_INVALID_ARGUMENT;
    std::unique_ptr<Model> m;
    try {
      m = std::make_unique<Model>(c, data);
    } catch (const std::runtime_error&) {
      return OR_INCOMPATIBLE;
    }
    models.insert(reinterpret_cast<or_model>(m.get()));
    ++object(c)->children;
    *out = reinterpret_cast<or_model>(m.release());
    return OR_OK;
  });
}

or_status describe(or_model m, or_model_info* out) {
  if (!models.count(m) || !out) return status(OR_INVALID_ARGUMENT);
  *out = {object(m)->id.c_str(), 1, 1,       object(m)->info_in,
          object(m)->info_out,   0, nullptr, nullptr};
  return OR_OK;
}

or_status create_session(or_model m, const or_session_options* o, or_session* out) {
  return protect([&]() -> or_status {
    if (!models.count(m) || !out) return OR_INVALID_ARGUMENT;
    if (o) {
      if (o->compile_policy != OR_ALLOW_COMPILE && o->compile_policy != OR_PRECOMPILED_ONLY)
        return OR_INVALID_ARGUMENT;
      if (o->shape_override_count > 1 || (o->shape_override_count && !o->shape_overrides))
        return OR_INVALID_ARGUMENT;
      if (o->shape_override_count &&
          (o->shape_overrides[0].input_index != 0 || !shape_ok(m, o->shape_overrides[0].shape, 0)))
        return OR_INVALID_ARGUMENT;
    }
    auto s = std::make_unique<Session>(m);
    sessions.insert(reinterpret_cast<or_session>(s.get()));
    ++object(m)->sessions;
    *out = reinterpret_cast<or_session>(s.release());
    return OR_OK;
  });
}

or_status requirements(or_session s, or_io_requirements* out) {
  if (!sessions.count(s) || !out) return status(OR_INVALID_ARGUMENT);
  *out = {1, 1, object(object(s)->model)->req_in, object(object(s)->model)->req_out};
  return OR_OK;
}

or_status import_buffer(or_context c, const or_memory_desc* d, or_buffer* out) {
  if (!contexts.count(c) || !d || !out) return status(OR_INVALID_ARGUMENT);
  return status(OR_UNSUPPORTED);
}

or_status allocate(or_context c, const or_allocation_desc* d, or_buffer* out) {
  return protect([&]() -> or_status {
    if (!contexts.count(c) || !d || !out) return OR_INVALID_ARGUMENT;
    if (!same(d->configuration) || d->vulkan) return OR_UNSUPPORTED;
    if (!d->byte_length || d->byte_length > SIZE_MAX || !d->byte_alignment ||
        (d->byte_alignment & (d->byte_alignment - 1)) || d->byte_alignment > SIZE_MAX ||
        d->access < OR_READ || d->access > OR_READ_WRITE)
      return OR_INVALID_ARGUMENT;
    auto b = std::make_unique<Buffer>();
    b->data = nullptr;
    if (posix_memalign(&b->data, std::max<uint64_t>(yolo26::kTensorAlignment, d->byte_alignment),
                       d->byte_length))
      return OR_OUT_OF_MEMORY;
    b->context = c;
    b->bytes = d->byte_length;
    b->access = d->access;
    buffers.insert(reinterpret_cast<or_buffer>(b.get()));
    ++object(c)->children;
    *out = reinterpret_cast<or_buffer>(b.release());
    return OR_OK;
  });
}

or_status buffer_info(or_buffer b, or_buffer_info* out) {
  if (!buffers.count(b) || !out) return status(OR_INVALID_ARGUMENT);
  *out = {OR_OWNER_RUNTIME,
          OR_ACCESSIBLE_HOST,
          OR_REFERENCE_POINTER,
          object(b)->access,
          object(b)->bytes,
          object(b)->data,
          nullptr};
  return OR_OK;
}
// Check a replacement binding completely before releasing the current valid binding.
or_status bind(or_session s, or_io io, uint32_t i, const or_tensor_binding* b) {
  if (!sessions.count(s)) return status(OR_INVALID_ARGUMENT);
  if (object(s)->running) return status(OR_BUSY);
  const int k = slot(io, i);
  if (k < 0) return status(OR_INVALID_ARGUMENT);
  if (!b) {
    unbind(s, static_cast<unsigned>(k));
    return OR_OK;
  }
  if (!buffers.count(b->buffer)) return status(OR_INVALID_ARGUMENT);
  if (!same(b->configuration)) return status(OR_UNSUPPORTED);
  const auto buf = b->buffer;
  if (object(buf)->context != object(object(s)->model)->context) return status(OR_INCOMPATIBLE);
  if (!shape_ok(object(s)->model, b->shape, static_cast<unsigned>(k), b->byte_strides))
    return status(OR_INVALID_ARGUMENT);
  if (b->byte_offset > object(buf)->bytes || b->byte_length > object(buf)->bytes - b->byte_offset ||
      b->byte_length < object(object(s)->model)->bytes[k])
    return status(OR_BUFFER_TOO_SMALL);
  if ((reinterpret_cast<uintptr_t>(object(buf)->data) + b->byte_offset) % yolo26::kTensorAlignment)
    return status(OR_INVALID_ARGUMENT);
  if ((k == 0 && !(object(buf)->access & OR_READ)) ||
      (k > 0 && !(object(buf)->access & OR_WRITE)))
    return status(OR_INCOMPATIBLE);
  const uintptr_t begin = reinterpret_cast<uintptr_t>(object(buf)->data) + b->byte_offset;
  for (unsigned j = 0; j < 2; ++j)
    if (j != static_cast<unsigned>(k) && object(s)->bound[j].buffer) {
      const uintptr_t other =
          reinterpret_cast<uintptr_t>(object(object(s)->bound[j].buffer)->data) +
          object(s)->bound[j].offset;
      if (begin < other + object(object(s)->model)->bytes[j] &&
          other < begin + object(object(s)->model)->bytes[k])
        return status(OR_INCOMPATIBLE);
    }
  unbind(s, static_cast<unsigned>(k));
  object(s)->bound[static_cast<unsigned>(k)] = {buf, b->byte_offset, b->byte_length};
  ++object(buf)->bound;
  return OR_OK;
}

or_status bind_host(or_session s, or_io io, uint32_t i, const or_host_tensor_binding* b) {
  if (!sessions.count(s) || slot(io, i) < 0) return status(OR_INVALID_ARGUMENT);
  if (object(s)->running) return status(OR_BUSY);
  if (!b) return bind(s, io, i, nullptr);
  return status(OR_UNSUPPORTED);
}

or_status validate(or_session s) {
  if (!sessions.count(s)) return status(OR_INVALID_ARGUMENT);
  if (object(s)->running) return status(OR_BUSY);
  for (const auto& b : object(s)->bound)
    if (!b.buffer) {
      object(s)->validated = false;
      return status(OR_STATE_INVALID);
    }
  object(s)->validated = true;
  return OR_OK;
}

or_status infer(or_session s, or_completion* out) {
  return protect([&]() -> or_status {
    std::unique_lock<std::recursive_mutex> lock(registry_mutex);
    if (!sessions.count(s) || !out) return OR_INVALID_ARGUMENT;
    if (object(s)->running) return OR_BUSY;
    if (!object(s)->validated || object(s)->invalid) return OR_STATE_INVALID;
    auto c = std::make_unique<Completion>();
    c->session = s;
    c->generation = object(s)->generation + 1;
    c->result = OR_OK;
    completions.insert(reinterpret_cast<or_completion>(c.get()));
    ++object(s)->generation;
    ++object(s)->completions;
    float* pointers[2];
    for (unsigned k = 0; k < 2; ++k)
      pointers[k] = reinterpret_cast<float*>(
          static_cast<uint8_t*>(object(object(s)->bound[k].buffer)->data) +
          object(s)->bound[k].offset);  // The running flag prevents rebinding/destruction while
                                        // compute releases the registry lock.
    object(s)->running = true;
    lock.unlock();
    try {
      if (y26_native_execute(object(s)->native.get(), pointers[0],
                             object(object(s)->model)->bytes[0], pointers[1],
                             object(object(s)->model)->bytes[1]) != Y26_NATIVE_OK)
        c->result = OR_EXECUTION_FAILED;
    } catch (...) {
      c->result = OR_EXECUTION_FAILED;
    }
    lock.lock();
    object(s)->running = false;
    if (c->result != OR_OK) object(s)->invalid = true;
    *out = reinterpret_cast<or_completion>(c.release());
    return OR_OK;
  });
}

or_status infer_async(or_session s, or_completion* out) {
  if (!sessions.count(s) || !out) return status(OR_INVALID_ARGUMENT);
  return status(OR_UNSUPPORTED);
}

or_status wait(or_completion c, uint64_t) {
  if (!completions.count(c)) return status(OR_INVALID_ARGUMENT);
  return status(object(c)->result);
}

or_status completion_info(or_completion c, or_completion_info* out) {
  if (!completions.count(c) || !out) return status(OR_INVALID_ARGUMENT);
  *out = {1, object(c)->result};
  return OR_OK;
}
// Generation checks reject output views invalidated by rebinding or a newer inference.
or_status outputs(or_session s, or_completion c, or_outputs* out) {
  if (!sessions.count(s) || !completions.count(c) || !out || object(c)->session != s ||
      object(c)->generation != object(s)->generation)
    return status(OR_INVALID_ARGUMENT);
  if (object(s)->running) return status(OR_BUSY);
  if (object(c)->result != OR_OK) return status(object(c)->result);
  for (unsigned j = 0; j < 1; ++j) {
    const unsigned k = j + 1;
    const auto& b = object(s)->bound[k];
    object(s)->outputs[j] = {
        j,
        {b.buffer,
         b.offset,
         object(object(s)->model)->bytes[k],
         {object(object(s)->model)->ranks[k], object(object(s)->model)->dims[k]},
         object(object(s)->model)->strides[k],
         config},
        nullptr};
    object(s)->output_ptrs[j] = &object(s)->outputs[j];
  }
  *out = {1, object(s)->output_ptrs};
  return OR_OK;
}

or_status reset(or_session s) {
  if (!sessions.count(s)) return status(OR_INVALID_ARGUMENT);
  if (object(s)->running) return status(OR_BUSY);
  object(s)->invalid = false;
  return OR_OK;
}

or_status release_completion(or_completion c) {
  if (!completions.count(c)) return status(OR_INVALID_ARGUMENT);
  --object(object(c)->session)->completions;
  completions.erase(c);
  delete object(c);
  return OR_OK;
}

or_status release_buffer(or_buffer b) {
  if (!buffers.count(b)) return status(OR_INVALID_ARGUMENT);
  if (object(b)->bound) return status(OR_BUSY);
  --object(object(b)->context)->children;
  buffers.erase(b);
  delete object(b);
  return OR_OK;
}

or_status destroy_session(or_session s) {
  if (!sessions.count(s)) return status(OR_INVALID_ARGUMENT);
  if (object(s)->running || object(s)->completions) return status(OR_BUSY);
  for (unsigned k = 0; k < 2; ++k) unbind(s, static_cast<unsigned>(k));
  --object(object(s)->model)->sessions;
  sessions.erase(s);
  delete object(s);
  return OR_OK;
}

or_status unload(or_model m) {
  if (!models.count(m)) return status(OR_INVALID_ARGUMENT);
  if (object(m)->sessions) return status(OR_BUSY);
  --object(object(m)->context)->children;
  models.erase(m);
  delete object(m);
  return OR_OK;
}

or_status destroy_context(or_context c) {
  if (!contexts.count(c)) return status(OR_INVALID_ARGUMENT);
  if (object(c)->children) return status(OR_BUSY);
  contexts.erase(c);
  delete object(c);
  return OR_OK;
}

or_status extension(or_context c, const char* id, const void** out) {
  if (!contexts.count(c) || !id || !out) return status(OR_INVALID_ARGUMENT);
  return status(OR_UNSUPPORTED);
}

const char* last_error() { return error; }
// Wrap registry-backed calls in a lock and exception boundary; infer manages its own lock.
template <auto F>
struct Locked;
template <typename... Args, or_status (*F)(Args...)>
struct Locked<F> {
  static or_status call(Args... args) noexcept {
    return protect([&]() -> or_status {
      std::lock_guard<std::recursive_mutex> lock(registry_mutex);
      return F(args...);
    });
  }
};
}  // namespace
extern "C" {
or_status OR_CALL y26_or_get_api(uint32_t requested, or_api* out) {
  if (requested != OR_RUNTIME_VERSION) return status(OR_INCOMPATIBLE);
  if (!out) return status(OR_INVALID_ARGUMENT);
  *out = {version,
          Locked<create_context>::call,
          Locked<capabilities>::call,
          Locked<load>::call,
          Locked<describe>::call,
          Locked<create_session>::call,
          Locked<requirements>::call,
          Locked<import_buffer>::call,
          Locked<allocate>::call,
          Locked<buffer_info>::call,
          Locked<bind>::call,
          Locked<bind_host>::call,
          Locked<validate>::call,
          infer,
          Locked<infer_async>::call,
          Locked<wait>::call,
          Locked<completion_info>::call,
          Locked<outputs>::call,
          Locked<reset>::call,
          Locked<release_completion>::call,
          Locked<release_buffer>::call,
          Locked<destroy_session>::call,
          Locked<unload>::call,
          Locked<destroy_context>::call,
          Locked<extension>::call,
          last_error};
  return OR_OK;
}
}
