#ifndef YOLO26N_FIXED_MEMORY_PLAN_H
#define YOLO26N_FIXED_MEMORY_PLAN_H

#include "fixed_schedule.h"

namespace yolo26::runtime_internal {

// Assigns every arena-resident tensor an offset, derives the arena and split-
// projection scratch sizes, and rejects a placement whose byte ranges overlap
// while both tensors are live. Tensor sizes must already be populated.
bool PlanScheduleMemory(ScheduleState* state);

// Rechecks bounds and live-range compatibility after placement. Exposed only
// to the native unit test; production initialization calls it through the
// planner above.
bool ValidateScheduleMemory(const ScheduleState& state);

}  // namespace yolo26::runtime_internal

#endif
