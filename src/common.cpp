#include "common.hpp"

int UV(int x, int y, int w) {
    return y*w+x;
}

void swap(float* a, float* b) {
    float* tmp;
    tmp = a;
    a = b;
    b = tmp;
}
