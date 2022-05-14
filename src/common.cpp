#include "common.hpp"

#include <sstream>
#include <iomanip>


void swap(float** a, float** b) {
    float* tmp = *a;
    *a = *b;
    *b = tmp;
}

std::string ftos(float v) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << v;
    return stream.str();
}

