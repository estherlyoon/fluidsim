#ifndef __FLUID_SIM_HPP__
#define __FLUID_SIM_HPP__

#include "common.hpp"

#include <vector>
#include <stdint.h>


class FluidSim {

public:
    // per-pixel scalar fields
    float* vx; // x-direction velocity field 
    float* vy; // y-direction velocity field
    float* pressures;
    float* temperatures;
    float* densities;
    uint8_t* RGBA;

    unsigned int width;
    unsigned int height;

    // time tracking
    TimePoint time;

    FluidSim(unsigned int w, unsigned int h);
    ~FluidSim();

    float updateTimestep();
};


#endif
