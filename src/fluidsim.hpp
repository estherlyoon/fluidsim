#ifndef __FLUID_SIM_HPP__
#define __FLUID_SIM_HPP__

#include "common.hpp"

#include <vector>
#include <stdint.h>

enum SmokeColor {
    WHITE,
    RED,
    GREEN,
    BLUE
};

class FluidSim {

public:
    bool gpu;

    // per-pixel scalar fields
    float* vx; // x-direction velocity field 
    float* vy; // y-direction velocity field
    float* temperatures; // TODO
    float* densities[3];
    uint8_t* RGBA;
    uint8_t* denseRGBA;
    uint8_t* cudaRGBA;
    uint8_t* cudaDenseRGBA;
    float* denseAdded[3];
    float* vxAdded;
    float* vyAdded;
    float* cudaDenseAdded;
    float* cudaVxAdded;
    float* cudaVyAdded;

    float* tmpV;
    float* tmpU;

    unsigned int width;
    unsigned int height;

    // forces applied by user
    int xPoint;
    int yPoint;

    // time tracking
    TimePoint time;

    // color
    float currColor[3];

    FluidSim(unsigned int w, unsigned int h, bool gpu);
    ~FluidSim();

    void updateSimulation();
    void reset();
    float updateTimestep();
    void addDensity(int x, int y);
    void addVelocity(int x, int y, float dx, float dy);
    void allocHost();
    void allocDevice();
    void changeColor(SmokeColor c);
    void setColor(float r, float g, float b);

};


#endif
