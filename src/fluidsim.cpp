#include "fluidsim.hpp"

#include <SFML/Graphics.hpp>

FluidSim::FluidSim(unsigned int w, unsigned int h) : width(w), height(h) {
    // assume fluid start with zero initial velocity and pressure
    vx = new float[w * h]();
    vy = new float[w * h]();
    pressures = new float[w * h]();
    temperatures = new float[w * h]();
    densities = new float[w * h]();
    // point (x, y) on screen -> (y*width+x)*4
    RGBA = new uint8_t[w * h * 4]();
    time = Time::now();

    // set opacity
    for (int i = 0; i < w*h; i++) {
        RGBA[i*4+3] = 255;
    }

}

FluidSim::~FluidSim() {

}

// returns time delta since last time recorded
float FluidSim::updateTimestep() {
    TimePoint currTS = this->time;
    auto t = Time::now();
    fsec fs = t - currTS;
    this->time = t;

    return fs.count();
}
