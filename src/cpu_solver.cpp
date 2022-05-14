#include "cpu_solver.hpp"
#include "common.cuh"

#include <cstring>
#include <iostream>

namespace cpu_solver {

void setBoundary(float* v, int w, int h, int bcond) {
    for (int i = 1; i < h; i++) {
        // vertical boundaries
        v[UV(0,i,w)] = bcond == CONTAINED_X ? -v[UV(1,i,w)] : v[UV(1,i,w)];
        v[UV(w-1,i,w)] = bcond == CONTAINED_X ? -v[UV(w-2,i,w)] : v[UV(w-2,i,w)];
    }

    for (int i = 1; i < w; i++) {
        // horizontal boundaries
        v[UV(i,0,w)] = bcond == CONTAINED_Y ? -v[UV(i,1,w)] : v[UV(i,1,w)];
        v[UV(i,h-1,w)] = bcond == CONTAINED_Y ? -v[UV(i,h-2,w)] : v[UV(i,h-2,w)];
    }

    v[UV(0,0,w)] = 0.5f * (v[UV(1,0,w)] + v[UV(0,1,w)]);
    v[UV(0,h-1,w)] = 0.5f * (v[UV(1,h-1,w)] + v[UV(0,h-2,w)]);
    v[UV(w-1,0,w)] = 0.5f * (v[UV(w-2,0,w)] + v[UV(w-1,1,w)]);
    v[UV(w-1,h-1,w)] = 0.5f * (v[UV(w-2,h-1,w)] + v[UV(w-1,h-2,w)]);
}

void updateBounds(float* v, int w, int h, uint8_t* bounds, bool isX) {
    for (int x = 1; x < w-1; x++) {
        for (int y = 1; y < h-1; y++) {
            int idx = UV(x,y,w);
            Boundary type = cellType(idx, bounds);
            switch(type) {
                case NORTH:
                    v[idx] = isX ? -v[UV(x,y-1,w)] : 0;
                    if (isX)
                        v[UV(x-1,y,w)] = -v[UV(x-1,y-1,w)];
                    break;
                case SOUTH:
                    v[idx] = isX ? -v[UV(x,y+1,w)] : 0;
                    if (isX)
                        v[UV(x+1,y,w)] = -v[UV(x+1,y+1,w)];
                    break;
                case EAST:
                    v[idx] = isX ? 0 : -v[UV(x+1,y,w)];
                    if (!isX)
                        v[UV(x,y-1,w)] = -v[UV(x+1,y-1,w)];
                    break;
                case WEST:
                    v[idx] = isX ? 0 : -v[UV(x-1,y,w)];
                    if (!isX)
                        v[UV(x,y+1,w)] = -v[UV(x-1,y+1,w)];
                    break;
                case NE:
                    v[idx] = 0;
                    if (isX)
                        v[UV(x-1,y,w)] = -v[UV(x-1,y-1,w)];
                    else
                        v[UV(x,y+1,w)] = -v[UV(x+1,y+1,w)];
                    break;
                case NW:
                    v[idx] = 0;
                    if (isX)
                        v[UV(x+1,y,w)] = -v[UV(x+1,y-1,w)];
                    else
                        v[UV(x,y+1,w)] = -v[UV(x-1,y+1,w)];
                    break;
                case SW:
                    v[idx] = 0;
                    if (isX)
                        v[UV(x+1,y,w)] = -v[UV(x+1,y+1,w)];
                    else
                        v[UV(x,y-1,w)] = -v[UV(x-1,y-1,w)];
                    break;
                case SE:
                    v[idx] = 0;
                    if (isX)
                        v[UV(x-1,y,w)] = -v[UV(x-1,y+1,w)];
                    else
                        v[UV(x,y-1,w)] = -v[UV(x+1,y-1,w)];
                    break;
                default:
                    break;    
            }
        }
    }
}

void advect(float* vx, float* vy, float* quantity, float* tmpQuantity, float timestep, unsigned int w, unsigned int h, int bcond, uint8_t* bounds) {

    // iterate through pixels
    for (int x = 1; x < w-1; x++) {
        for (int y = 1; y < h-1; y++) {
            // trace current cell backwards
            int idx = UV(x, y, w);
            float xPos = (float)x - timestep * vx[idx];
            float yPos = (float)y - timestep * vy[idx];

            // ignore OOB positions
            if (xPos < 0.5f || xPos > w-1.5f || yPos < 0.5f || yPos > h-1.5f)
                continue;
            if (isCenter(idx, bounds))
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

            // set quantity with interpolated values
            float lastQ = quantity[idx];
            quantity[idx] = b1*(a1*tmpQuantity[UV(x0,y0,w)] + a0*tmpQuantity[UV(x1,y0,w)])
                        + b0*(a1*tmpQuantity[UV(x0,y1,w)] + a0*tmpQuantity[UV(x1,y1,w)]);
            
        }
    }
    
    setBoundary(quantity, w, h, bcond);
    if (bcond == CONTAINED_X || bcond == CONTAINED_Y) {
        updateBounds(quantity, w, h, bounds, bcond == CONTAINED_X);
    }
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

void diffuse(float* densities, float* tmpDensities, float diff_rate, int w, int h, float ts, int iters, int bcond, uint8_t* bounds) {
    float a = ts * diff_rate * w * h;

    for (int k = 0; k < iters; k++) {
        for (int x = 1; x < w-1; x++) {
            for (int y = 1; y < h-1; y++) {
                if (isCenter(UV(x,y,w), bounds))
                    continue;
                densities[UV(x,y,w)] = (tmpDensities[UV(x,y,w)] +
                                a * (tmpDensities[UV(x,y-1,w)] + tmpDensities[UV(x,y+1,w)]
                                     + tmpDensities[UV(x-1,y,w)] + tmpDensities[UV(x+1,y,w)]))
                                    / (1 + 4 * a);
                
            }
        }
        setBoundary(densities, w, h, bcond);
    }
}

void project(float* vx, float* vy, float* p, float* div, int w, int h, int iters, uint8_t* bounds) {
    int dim = w * h;
    float h0 = 1.0f / h;
    float w0 = 1.0f / w;

    for (int x = 1; x < w-1; x++) {
        for (int y = 1; y < h-1; y++) {
            int idx = y * w + x;
            if (isCenter(idx, bounds))
                continue;
            div[idx] = -0.5f*h0*(vy[UV(x,y+1,w)] - vy[UV(x,y-1,w)]
                                 + vx[UV(x+1,y,w)] - vx[UV(x-1,y,w)]);
            p[idx] = 0;
        }
    }
    setBoundary(p, w, h, CONTINUOUS);
    setBoundary(div, w, h, CONTINUOUS);

    // set bnd for div, p
    for (int i = 0; i < iters; i++) {
        for (int x = 1; x < w-1; x++) {
            for (int y = 1; y < h-1; y++) {
                int idx = y * w + x;
                if (isCenter(idx, bounds))
                    continue;
                p[idx] = (div[idx] + p[UV(x-1,y,w)] + p[UV(x+1,y,w)]
                                + p[UV(x,y-1,w)] + p[UV(x,y+1,w)]) / 4.0f;
            }
        }
        setBoundary(p, w, h, CONTINUOUS);
    }

    for (int x = 1; x < w-1; x++) {
        for (int y = 1; y < h-1; y++) {
            int idx = y * w + x;
            if (isCenter(idx, bounds))
                continue;
            vx[idx] -= 0.5 * (p[UV(x+1,y,w)] - p[UV(x-1,y,w)]) / w0;
            vy[idx] -= 0.5 * (p[UV(x,y+1,w)] - p[UV(x,y-1,w)]) / h0;
        }
    }
    setBoundary(vx, w, h, CONTAINED_X);
    updateBounds(vx, w, h, bounds, true);
    setBoundary(vy, w, h, CONTAINED_Y);
    updateBounds(vy, w, h, bounds, false);
}

void updateColors(float* densities, uint8_t* res, int w, int h, int color, uint8_t* bounds) {
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            if (isCenter(UV(x,y,w), bounds))
                continue;
            float density = densities[UV(x,y,w)];
            res[UV(x,y,w)*4+color] = density * 255.0f;
        }
    }
}

void solveDensity(FluidSim* sim, int c) {
    float diff_rate = 0.8; // TODO
    addSources(sim->denseAdded[c], sim->densities[c], sim->width, sim->height, sim->timeDelta, 1.0f);

    swap(&sim->densities[c], &sim->tmpV);
    diffuse(sim->densities[c], sim->tmpV, diff_rate, sim->width, sim->height, sim->timeDelta, 20, CONTINUOUS, sim->bounds);
    swap(&sim->densities[c], &sim->tmpV);
    advect(sim->vx, sim->vy, sim->densities[c], sim->tmpV, sim->timeDelta, sim->width, sim->height, CONTINUOUS, sim->bounds);
    updateColors(sim->densities[c], sim->denseRGBA, sim->width, sim->height, c, sim->bounds);
}

void solveVelocity(FluidSim* sim) {
    addSources(sim->vxAdded, sim->vx, sim->width, sim->height, sim->timeDelta, 0.0f);
    addSources(sim->vyAdded, sim->vy, sim->width, sim->height, sim->timeDelta, 0.0f);
    addSources(sim->tempAdded, sim->vy, sim->width, sim->height, sim->timeDelta, 0.0f);

    swap(&sim->vx, &sim->tmpV);
    diffuse(sim->vx, sim->tmpV, sim->viscosity, sim->width, sim->height, sim->timeDelta, 20, CONTAINED_X, sim->bounds);
    swap(&sim->vy, &sim->tmpV);
    diffuse(sim->vy, sim->tmpV, sim->viscosity, sim->width, sim->height, sim->timeDelta, 20, CONTAINED_Y, sim->bounds);

    project(sim->vx, sim->vy, sim->tmpV, sim->tmpU, sim->width, sim->height, 40, sim->bounds);

    swap(&sim->vx, &sim->tmpV);
    advect(sim->vx, sim->vy, sim->vx, sim->tmpV, sim->timeDelta, sim->width, sim->height, CONTAINED_X, sim->bounds);
    swap(&sim->vy, &sim->tmpV);
    advect(sim->vx, sim->vy, sim->vy, sim->tmpV, sim->timeDelta, sim->width, sim->height, CONTAINED_Y, sim->bounds);

    project(sim->vx, sim->vy, sim->tmpV, sim->tmpU, sim->width, sim->height, 40, sim->bounds);
}


void update(FluidSim* sim) {
    solveVelocity(sim);
    solveDensity(sim, 0);
    solveDensity(sim, 1);
    solveDensity(sim, 2);
}


}
