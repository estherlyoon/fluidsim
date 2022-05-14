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
    float* temperatures;
    float* tempAdded;
    float* densities[3];
    uint8_t* RGBA;
    uint8_t* denseRGBA;
    uint8_t* cudaDenseRGBA;
    float* denseAdded[3];
    float* vxAdded;
    float* vyAdded;
    float* cudaDenseAdded[3];
    float* cudaVxAdded;
    float* cudaVyAdded;
    float* cudaTempAdded;
    uint8_t* bounds;
    uint8_t* cudaBounds;

    float* tmpV;
    float* tmpU;

    unsigned int width;
    unsigned int height;

    // forces applied by user
    int xPoint;
    int yPoint;

    float timeDelta;
    float viscosity;
    float smokeSize;

    // time tracking
    TimePoint time;

    // color
    float currColor[3];

    // temperature
    float kd; // mass scale factor * smoke density
    float tempDelta; // ambient temperature difference

    FluidSim(unsigned int w, unsigned int h, bool gpu);
    ~FluidSim();

    void updateSimulation();
    void reset();
    float updateTimestep();
    void addBoundary(int x, int y);
    void addDensity(int x, int y);
    void addVelocity(int x, int y, float dx, float dy);
    void allocHost();
    void allocDevice();
    void changeColor(SmokeColor c);
    void setColor(float r, float g, float b);

};


#endif
