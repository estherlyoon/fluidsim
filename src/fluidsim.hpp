#ifndef __FLUID_SIM_HPP__
#define __FLUID_SIM_HPP__

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

    FluidSim(unsigned int w, unsigned int h);
    ~FluidSim();
};


#endif
