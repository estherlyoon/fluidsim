#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <chrono>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;
typedef std::chrono::time_point<Time> TimePoint;

int UV(int x, int y, int w);
void swap(float* a, float* b);

// boundary conditions
#define CONTAINED_X 0
#define CONTAINED_Y 1
#define CONTINUOUS 2

#endif
