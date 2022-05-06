#include "cpu_solver.hpp"

#include <cstring>
#include <iostream>

namespace cpu_solver {


void advect(float* vx, float* vy, uint8_t* RGBA, float timestep, unsigned int w, unsigned int h) {

    const int dim = w * h;
    // scaling factor for velocities
    const float rdx = 1.0/dim;
    // make tmp arrays, store result in vx and vy
    float* currVx = new float[dim];
    float* currVy = new float[dim];
    uint8_t* currRGBA = new uint8_t[dim*4];
    std::memcpy(currVx, vx, sizeof(float) * dim);
    std::memcpy(currVy, vy, sizeof(float) * dim);
    std::memcpy(currRGBA, RGBA, sizeof(uint8_t) * dim * 4);

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

            // set vx and vy with interpolated values
            vx[idx] = b1*(a1*currVx[y0*w+x0] + a0*currVx[y0*w+x1])
                        + b0*(a1*currVx[y1*w+x0] + a0*currVx[y1*w+x1]);
            vy[idx] = b1*(a1*currVy[y0*w+x0] + a0*currVy[y0*w+x1])
                        + b0*(a1*currVy[y1*w+x0] + a0*currVy[y1*w+x1]);

            // interpolate colored dye
            for (int i = 0; i < 3; i++) {
                RGBA[idx * 4 + i] = b1*(a1*currRGBA[(y0*w+x0)*4+i] + a0*currRGBA[(y0*w+x1)*4+i])
                        + b0*(a1*currRGBA[(y1*w+x0)*4+i] + a0*currRGBA[(y1*w+x1)*4+i]);
            }
        }
    }

    // TODO optimize
    delete currVx;
    delete currVy;
}

// TODO how to make general-- pass in some grid with parameterized type instead of two vecs? or just pass in one vec at a time?
// template type for in>
void jacobi(float* xIn, float* bIn, float alpha, float rBeta, int w, int h, int iters) {

    uint8_t* currIn = new uint8_t[w*h];
    std::memcpy(currIn, xIn, sizeof(uint8_t) * w * h);

    // perform jacobi iterations 
    for (int i = 0; i < iters; i++) {
        for (int x = 1; x < w-1; x++) {
            for (int y = 1; y < h-1; y++) { 
                float xDown = currIn[(y-1)*w+x];
                float xUp = currIn[(y+1)*w+x];
                float xRight = currIn[y*w+x+1];
                float xLeft = currIn[y*w+x-1];

                float bSample = bIn[y*w+x];

                xIn[y*w+x] = (xDown + xUp + xLeft + xRight + alpha * bSample) * rBeta;
            }
        }
    }

    delete currIn;
}

void update(FluidSim* sim) {
    // get timestep
    float timestep = sim->updateTimestep();
    //printf("timestep = %f\n", timestep);
    float squaredDim = sim->width*sim->width*sim->height*sim->height;
    const float alpha = squaredDim/timestep;
    const float beta = 1/(4 + squaredDim/timestep);

    // update advected quantities (velocities, color, TODO density/pressure)
    advect(sim->vx, sim->vy, sim->RGBA, 1.0, sim->width, sim->height);
    // TODO alpha beta
    jacobi(sim->vx, sim->vx, alpha, beta, sim->width, sim->height, 35);
    jacobi(sim->vy, sim->vy, alpha, beta, sim->width, sim->height, 35);
}


}
