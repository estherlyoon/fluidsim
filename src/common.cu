#include "common.cuh"

__host__ __device__ int UV(int x, int y, int w) {
    return y*w+x;
}

void swap(float** a, float** b) {
    float* tmp = *a;
    *a = *b;
    *b = tmp;
}
