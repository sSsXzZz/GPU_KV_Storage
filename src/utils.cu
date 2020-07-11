#include "utils.h"

namespace {

// Statically allocate timespec used in all get_time calls
struct timespec tmp_timespec;

}  // namespace

time_t get_time_us() {
    clock_gettime(CLOCK_MONOTONIC, &tmp_timespec);
    return (tmp_timespec.tv_sec * 1'000'000) + (tmp_timespec.tv_nsec / 1000);
}
