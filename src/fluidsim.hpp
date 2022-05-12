#ifndef __FLUID_SIM_HPP__
#define __FLUID_SIM_HPP__

#include "common.cuh"

#include <vector>
#include <stdint.h>


class FluidSim {

public:
    bool gpu;

    // per-pixel scalar fields
    float* vx; // x-direction velocity field 
    float* vy; // y-direction velocity field
    float* pressures; // TODO
    float* temperatures; // TODO
    float* densities;
    uint8_t* RGBA;
    uint8_t* denseRGBA;
    float* denseAdded;
    float* vxAdded;
    float* vyAdded;

    float* tmpV;
    float* tmpU;

    unsigned int width;
    unsigned int height;

    // forces applied by user
    int xPoint;
    int yPoint;

    // time tracking
    TimePoint time;

    FluidSim(unsigned int w, unsigned int h, bool gpu);
    ~FluidSim();

    void updateSimulation();
    float updateTimestep();
    void addDensity(int x, int y);
    void addVelocity(int x, int y, float dx, float dy);
    void allocHost();
    void allocDevice();

};


#endif
