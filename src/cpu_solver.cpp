#include "cpu_solver.hpp"

#include <cstring>
#include <iostream>

namespace cpu_solver {

void advect(float* vx, float* vy, float* quantity, float timestep, unsigned int w, unsigned int h) {

    const int dim = w * h;

    // make tmp array for old values
    float* currQuant = new float[dim];
    std::memcpy(currQuant, quantity, sizeof(float) * dim);

    // iterate through pixels
    for (int x = 1; x < w-1; x++) {
        for (int y = 1; y < h-1; y++) {
            // trace current cell backwards
            int idx = y * w + x;
            /* std::cout << "idx = " << idx << std::endl; */
            float xPos = (float)x - timestep * vx[idx];
            float yPos = (float)y - timestep * vy[idx];

            // ignore OOB positions
            if (xPos < 0.5f || xPos > w-1.5f || yPos < 0.5f || yPos > h-1.5f)
                continue;

            // interpolate four points closest to position
            int x0 = static_cast<int>(xPos);
            int y0 = static_cast<int>(yPos);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float a0 = xPos - (float)x0;
            float a1 = 1.0f - a0;
            float b0 = yPos - (float)y0;
            float b1 = 1.0f - b0;

            if (vx[idx] != 0.0f) {
                /* printf("idx = (%d, %d)\n", x, y); */
                /* printf("vx[%d] = %f\n", idx, vx[idx]); */
                /* printf("x0 = %d\n", x0); */
                /* printf("\tx: %f/%f of %d/%d\n", a0, a1, x1, x0); */
                /* printf("\ty: %f/%f of %d/%d\n", b0, b1, y1, y0); */
                /* printf("\t\tmain term = %f\n", currQuant[y0*w+x0]); */
            }
            
            // set quantity with interpolated values
            float lastQ = quantity[idx];
            quantity[idx] = b1*(a1*currQuant[y0*w+x0] + a0*currQuant[y0*w+x1])
                        + b0*(a1*currQuant[y1*w+x0] + a0*currQuant[y1*w+x1]);

            /* if (lastQ < quantity[idx]) */
            /*     printf("quant[%d] = %f -> %f\n", idx, lastQ, quantity[idx]); */

            
        }
    }

    // TODO optimize
    delete currQuant;
}

// TODO how to make general-- pass in some grid with parameterized type instead of two vecs? or just pass in one vec at a time?
// template type for in>
void jacobi(float* xIn, float* bIn, float alpha, float rBeta, int w, int h, int iters) {

    float* currIn = new float[w*h];
    std::memcpy(currIn, xIn, sizeof(float) * w * h);

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

void addSources(float* src, float* dest, int w, int h, float ts, float clamp) {
    for (int i = 0; i < w*h; i++) {
        if (clamp)
            dest[i] = std::min(clamp, dest[i] + ts*src[i]);
        else
            dest[i] += ts*src[i];
        src[i] = 0;
    }
}

void diffuse(float* densities, float* tmpDensities, float diff_rate, int w, int h, float ts) {
    float a = ts * diff_rate * w * h;

    for (int k = 0; k < 20; k++) {
        for (int x = 1; x < w-1; x++) {
            for (int y = 1; y < h-1; y++) {
                int idx = y * w + x;
                densities[idx] = (tmpDensities[idx] +
                                a * (tmpDensities[(y-1)*w+x] + tmpDensities[(y+1)*w+x]
                                     + tmpDensities[y*w+x-1] + tmpDensities[y*w+x+1]))
                                    / (1 + 4 * a);
                
            }
        }
    }
}

void updateColors(float* densities, uint8_t* RGBA, uint8_t* res, int w, int h) {
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int idx = y * w + x;

            float density = densities[idx];
            for (int c = 0; c < 3; c++) {
                res[idx*4+c] = density * RGBA[idx*4+c];
            }

        }
    }
}

void solveDensity(FluidSim* sim) {
    float diff_rate = 0.5; // TODO
    // TODO timestep?
    addSources(sim->denseAdded, sim->densities, sim->width, sim->height, 1.0, 1.0f);

    int dim = sim->width * sim->height;
    float* tmpDensities = new float[dim];
    std::memcpy(tmpDensities, sim->densities, sizeof(float) * dim);

    diffuse(sim->densities, tmpDensities, diff_rate, sim->width, sim->height, 1.0);
    advect(sim->vx, sim->vy, sim->densities, 1.0, sim->width, sim->height);
    updateColors(sim->densities, sim->RGBA, sim->denseRGBA, sim->width, sim->height);
}

void solveVelocity(FluidSim* sim) {
    addSources(sim->vxAdded, sim->vx, sim->width, sim->height, 1.0f, 0.0f);
    addSources(sim->vyAdded, sim->vy, sim->width, sim->height, 1.0f, 0.0f);

    int dim = sim->width * sim->height;
    float visc = 0.8;
    float* tmpV = new float[dim];
    std::memcpy(tmpV, sim->vx, sizeof(float) * dim);
    diffuse(sim->vx, tmpV, visc, sim->width, sim->height, 1.0); // TODO time?
    std::memcpy(tmpV, sim->vy, sizeof(float) * dim);
    diffuse(sim->vy, tmpV, visc, sim->width, sim->height, 1.0); // TODO time?

    // project
    /* jacobi(sim->vx, sim->vx, alpha, beta, sim->width, sim->height, 35); */
    /* jacobi(sim->vy, sim->vy, alpha, beta, sim->width, sim->height, 35); */

    advect(sim->vx, sim->vy, sim->vx, 1.0, sim->width, sim->height);
    advect(sim->vx, sim->vy, sim->vy, 1.0, sim->width, sim->height);
}


void update(FluidSim* sim, float timestep) {
    solveDensity(sim);
    solveVelocity(sim);
}


}
