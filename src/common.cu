#include "common.cuh"


void swap(float** a, float** b) {
    float* tmp = *a;
    *a = *b;
    *b = tmp;
}
