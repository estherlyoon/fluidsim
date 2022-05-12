#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda.h>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;
typedef std::chrono::time_point<Time> TimePoint;

inline __host__ __device__ int UV(int x, int y, int w) {
    return y*w+x;
}

void swap(float** a, float** b);

// boundary conditions
#define CONTAINED_X 0
#define CONTAINED_Y 1
#define CONTINUOUS 2

#endif
