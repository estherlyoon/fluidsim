#include "fluidsim.hpp"
#include "cpu_solver.hpp"

#include <SFML/Graphics.hpp>

FluidSim::FluidSim(unsigned int w, unsigned int h, bool gpu) : width(w), height(h), gpu(gpu), addVelocity(false) {
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

void FluidSim::updateSimulation() {
    float timestep = updateTimestep();

    if (addVelocity) {
        // scale xDir and yDir some amount add add to vx and vy
        // TODO later make more complex with force equation to include impulse?
    }

    if (gpu) {

    } else {
        cpu_solver::update(this, timestep);
    }
}

// returns time delta since last time recorded
float FluidSim::updateTimestep() {
    TimePoint currTS = this->time;
    auto t = Time::now();
    fsec fs = t - currTS;
    this->time = t;

    return fs.count();
}
