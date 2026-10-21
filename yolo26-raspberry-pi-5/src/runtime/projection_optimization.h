#ifndef YOLO26N_PROJECTION_OPTIMIZATION_H
#define YOLO26N_PROJECTION_OPTIMIZATION_H

#include <pthreadpool.h>
#include <xnnpack.h>

#include <cstddef>
#include <cstdint>

// Two early 1x1 projections normally produce a combined tensor that must then
// be transposed before its two output-channel halves enter the next C2f block.
// This adapter stops the first two graphs at those projections and invokes each
// packed XNNPACK operator twice: once for each output-channel half. Each call
// writes directly in the layout consumed by the next graph, avoiding the two
// combined intermediate tensors and their transposes. Both projections use the
// same scratch allocation because the P2 halves are no longer needed when the
// P3 projection begins.
struct ProjectionGraphs {
  xnn_runtime_t stem_to_p2;
  xnn_runtime_t p2_to_p3;
  xnn_runtime_t p3_to_p4;
};

struct ProjectionDimensions {
  size_t p2_half_elements;
  size_t p3_half_elements;
  size_t p2_pixels;
  size_t p3_pixels;
};

struct ProjectionBuffers {
  uint16_t* scratch;
  uint16_t* input;
  uint16_t* backbone_p3;
  uint16_t* p4_entry;
};

// Marks the intermediate hand-off values as external. Call once after graph
// construction and before setup or execution.
bool ConfigureProjectionOptimization(const ProjectionGraphs& graphs);

// Executes the three early-backbone graphs, pausing at the two projections to
// write their output-channel halves directly. All buffers are FP16. Scratch
// holds both halves of the larger projection and must provide the
// ScheduleState::project_bytes extent computed by the fixed schedule.
bool RunProjectionOptimization(const ProjectionGraphs& graphs, pthreadpool_t threadpool,
                               const ProjectionDimensions& dimensions,
                               const ProjectionBuffers& buffers);

#endif
