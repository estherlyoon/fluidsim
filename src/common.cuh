#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#include <chrono>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda.h>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;
typedef std::chrono::time_point<Time> TimePoint;

enum Boundary {
    NORTH,
    EAST,
    SOUTH,
    WEST,
    NE,
    NW,
    SE,
    SW,
    CENTER,
    FLUID,
    NONE
};

inline __device__ __host__ int UV(int x, int y, int w) {
    return y*w+x;
}

inline __device__ __host__ void setCenter(int idx, uint8_t* bounds) {
    bounds[idx] |= 16;
}

inline __device__ __host__ void setEast(int idx, uint8_t* bounds) {
    bounds[idx] |= 8;
}

inline __device__ __host__ void setWest(int idx, uint8_t* bounds) {
    bounds[idx] |= 4;
}

inline __device__ __host__ void setSouth(int idx, uint8_t* bounds) {
    bounds[idx] |= 2;
}

inline __device__ __host__ void setNorth(int idx, uint8_t* bounds) {
    bounds[idx] |= 1;
}

inline __device__ __host__ bool isCenter(int idx, uint8_t* bounds) {
    return bounds[idx] & 16;
}

inline __device__ __host__ bool isEast(int idx, uint8_t* bounds) {
    return bounds[idx] & 8;
}

inline __device__ __host__ bool isWest(int idx, uint8_t* bounds) {
    return bounds[idx] & 4;
}

inline __device__ __host__ bool isSouth(int idx, uint8_t* bounds) {
    return bounds[idx] & 2;
}

inline __device__ __host__ bool isNorth(int idx, uint8_t* bounds) {
    return bounds[idx] & 1;
}


__device__ __host__ Boundary cellType(int idx, uint8_t* bounds);
void setCellType(Boundary b, int idx, uint8_t* bounds);
void swap(float** a, float** b);
std::string ftos(float v);

// boundary conditions
#define CONTAINED_X 0
#define CONTAINED_Y 1
#define CONTINUOUS 2

#endif
