#include "fluidsim.hpp"

#include <SFML/Graphics.hpp>

FluidSim::FluidSim(unsigned int w, unsigned int h) {
    vx = new float[w * h]();
    vy = new float[w * h]();
    pressures = new float[w * h]();
    temperatures = new float[w * h]();
    densities = new float[w * h]();
    RGBA = new uint8_t[w * h * 4]();

    // set opacity
    for (int i = 0; i < w*h; i++) {
        RGBA[i*4+3] = 255;
    }

}

FluidSim::~FluidSim() {

}
