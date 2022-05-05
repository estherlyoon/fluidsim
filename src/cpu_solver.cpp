#include "cpu_solver.hpp"

#include <cstring>
#include <iostream>

namespace cpu_solver {


void advect(float* vx, float* vy, float timestep, unsigned int w, unsigned int h) {

    const int dim = w * h;
    // scaling factor for velocities
    const float rdx = 1.0/dim;
    // make tmp arrays, store result in vx and vy
    float* currVx = new float[dim];
    float* currVy = new float[dim];
    std::memcpy(currVx, vx, sizeof(float) * dim);
    std::memcpy(currVy, vy, sizeof(float) * dim);

    // iterate through pixels
    for (int x = 1; x < w-1; x++) {
        for (int y = 1; y < h-1; y++) {
            // trace current cell backwards
            int idx = y * w + x;
            /* std::cout << "idx = " << idx << std::endl; */
            int xPos = x - timestep * rdx * currVx[idx];
            int yPos = y - timestep * rdx * currVy[idx];

            // ignore OOB positions
            if (xPos < 0.5f || xPos > w-1.5f || yPos < 0.5f || yPos > h-1.5f)
                continue;

            // interpolate four points closest to position
            int x0 = static_cast<int>(xPos);
            int y0 = static_cast<int>(yPos);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float a0 = xPos - x0;
            float a1 = 1 - a0;
            float b0 = yPos - y0;
            float b1 = 1 - b0;

            /* std::cout << "i0 = " << y0*w+x0 << std::endl; */
            /* std::cout << "i1 = " << y0*w+x1 << std::endl; */
            /* std::cout << "i2 = " << y1*w+x0 << std::endl; */
            /* std::cout << "i3 = " << y1*w+x1 << std::endl; */

            // set vx and vy with interpolated values
            vx[idx] = b1*(a1*currVx[y0*w+x0] + a0*currVx[y0*w+x1])
                        + b0*(a1*currVx[y1*w+x0] + a0*currVx[y1*w+x1]);
            vy[idx] = b1*(a1*currVy[y0*w+x0] + a0*currVy[y0*w+x1])
                        + b0*(a1*currVy[y1*w+x0] + a0*currVy[y1*w+x1]);
        }
    }

    // TODO optimize
    /* delete currVx; */
    /* delete currVy; */
}

void update(FluidSim* sim) {
    advect(sim->vx, sim->vy, 1.0, sim->width, sim->height);
}


}
