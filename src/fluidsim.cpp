#include "fluidsim.hpp"
#include "cpu_solver.hpp"

#include <SFML/Graphics.hpp>

FluidSim::FluidSim(unsigned int w, unsigned int h, bool gpu) : width(w), height(h), gpu(gpu), xPoint(0), yPoint(0) {
    // assume fluid start with zero initial velocity and pressure
    vx = new float[w * h]();
    vy = new float[w * h]();
    pressures = new float[w * h]();
    temperatures = new float[w * h]();
    densities = new float[w * h]();
    denseAdded = new float[w * h]();
    vxAdded = new float[w * h]();
    vyAdded = new float[w * h]();
    // point (x, y) on screen -> (y*width+x)*4
    RGBA = new uint8_t[w * h * 4]();
    denseRGBA = new uint8_t[w * h * 4]();
    time = Time::now();

    // set opacity
    for (int i = 0; i < w*h; i++) {
        // init base color to white
        for (int j = 0; j < 4; j++) 
            RGBA[i*4+j] = 255;
        denseRGBA[i*4+3] = 255;
    }

}

FluidSim::~FluidSim() {

}

void FluidSim::updateSimulation() {
    float timestep = updateTimestep();

    if (gpu) {

    } else {
        cpu_solver::update(this, timestep);
    }
}

void FluidSim::addDensity(int x, int y) {
    for (int i = -2; i < 3; i++) {
        for (int j = -2; j < 3; j++) {
            int idx = (y + i) * width + x + j;
            if (idx < 0 || idx >= width * height)
                continue;

            denseAdded[idx] = 1.0;

            for (int k = 0; k < 3; k++) {
                // TODO change to adding whatever color user has set
                denseRGBA[idx*4+k] = 255;
                RGBA[idx*4+k] = 255;
            }
        }
    }
}

void FluidSim::addVelocity(int x, int y, float dx, float dy) {
    int idx = y * width + x;
    /* printf("adding %f to x = %d, %f to y = %d\n", dx, x, -1*dy, y); */
    vxAdded[idx] += 0.5f * width * dx;
    vyAdded[idx] += 0.5f * height * -1*dy;
}

// returns time delta since last time recorded
float FluidSim::updateTimestep() {
    TimePoint currTS = this->time;
    auto t = Time::now();
    fsec fs = t - currTS;
    this->time = t;

    return fs.count();
}
