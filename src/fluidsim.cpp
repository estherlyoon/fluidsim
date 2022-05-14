#include "fluidsim.hpp"
#include "cpu_solver.hpp"
#include "gpu_solver.cuh"

#include <SFML/Graphics.hpp>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cuda.h>

FluidSim::FluidSim(unsigned int w, unsigned int h, bool gpu) : width(w), height(h), gpu(gpu), xPoint(0), yPoint(0), kd(2.0), timeDelta(1.0), viscosity(1.0), smokeSize(3.0), tempDelta(1.0) {
    vxAdded = new float[w * h]();
    vyAdded = new float[w * h]();
    denseRGBA = new uint8_t[w * h * 4]();
    tempAdded = new float[w * h]();
    bounds = new uint8_t[w * h]();

    for (int i = 0; i < 3; i++) {
        denseAdded[i] = new float[w * h]();
        currColor[i] = 1.0;
    }

    if (gpu) allocDevice();
    else allocHost();
    
    // set opacity
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int i = y * w + x;
            // init base color to white
            for (int j = 0; j < 4; j++) 
            if (x > 0 && y > 0 && x < w-1 && y < h-1 && (i > 12800 && i < 25600) || (i < 102400 && i > 80000)) {
                if (x < w/3) {
                    denseRGBA[i*4+0] = 255;
                    denseAdded[0][i] = 1.0;
                }
                else if (x > w/3 && x < 2*w/3) {
                    denseRGBA[i*4+1] = 255;
                    denseAdded[1][i] = 1.0;
                }
                else if (x > 2*w/3) {
                    denseRGBA[i*4+2] = 255;
                    denseAdded[2][i] = 1.0;
                }
                vyAdded[i] = 1.0;
                vxAdded[i] = 1.0;
            }
            denseRGBA[i*4+3] = 255;
            tempAdded[i] -= 2.0;
        }
    }
}

FluidSim::~FluidSim() {
 // TODO cuda free
}

void FluidSim::reset() {
    size_t dim = sizeof(float) * width * height;
    memset(denseRGBA, 0, sizeof(uint8_t) * 4 * width * height);

    if (gpu) {
        cudaMemset(vx, 0, dim);
        cudaMemset(vy, 0, dim);
        cudaMemset(densities[0], 0, dim);
        cudaMemset(densities[1], 0, dim);
        cudaMemset(densities[2], 0, dim);
        cudaMemset(cudaBounds, 0, sizeof(uint8_t) * width * height);
    } else {
        memset(vx, 0, dim);
        memset(vy, 0, dim);
        memset(densities[0], 0, dim);
        memset(densities[1], 0, dim);
        memset(densities[2], 0, dim);
    }

    memset(tempAdded, 0, dim);
    memset(vxAdded, 0, dim);
    memset(vyAdded, 0, dim);
    memset(denseAdded[0], 0, dim);
    memset(denseAdded[1], 0, dim);
    memset(denseAdded[2], 0, dim);
    memset(bounds, 0, sizeof(uint8_t) * width * height);

    for (int i = 0; i < width*height; i++) {
        denseRGBA[i*4+3] = 255;
    }
}

void FluidSim::allocHost() {
    size_t dim = width * height;
    // assume fluid starts with zero initial velocity and temperature
    vx = new float[dim]();
    vy = new float[dim]();
    temperatures = new float[dim]();
    for (int i = 0; i < 3; i++)
        densities[i] = new float[dim]();

    // point (x, y) on screen -> (y*width+x)*4
    tmpV = new float[dim];
    tmpU = new float[dim];
}

void FluidSim::allocDevice() {
    size_t dim = sizeof(float) * width * height;
    cudaMalloc((void**)&vx, dim);
    cudaMalloc((void**)&vy, dim);
    cudaMalloc((void**)&temperatures, dim);
    cudaMalloc((void**)&tmpV, dim);
    cudaMalloc((void**)&tmpU, dim);
    cudaMalloc((void**)&cudaVxAdded, dim);
    cudaMalloc((void**)&cudaVyAdded, dim);
    cudaMalloc((void**)&cudaTempAdded, dim);
    cudaMalloc((void**)&cudaDenseRGBA, sizeof(uint8_t)*width*height*4);
    cudaMalloc((void**)&cudaBounds, sizeof(uint8_t)*width*height);

    for (int i = 0; i < 3; i++) {
        cudaMalloc((void**)&cudaDenseAdded[i], dim);
        cudaMalloc((void**)&densities[i], dim);
    }

    // init to 0
    cudaMemset(vx, 0, dim);
    cudaMemset(vy, 0, dim);
    cudaMemset(tmpV, 0, dim);
    cudaMemset(tmpU, 0, dim);
    cudaMemset(cudaDenseRGBA, 255, sizeof(uint8_t)*width*height);
    for (int i = 0; i < 3; i++)
        cudaMemset(densities[i], 0, dim);
}
 
void FluidSim::setColor(float r, float g, float b) {
    currColor[0] = r;
    currColor[1] = g;
    currColor[2] = b;
}

void FluidSim::changeColor(SmokeColor c) {
    switch(c) {
        case SmokeColor::WHITE:
            setColor(1.0, 1.0, 1.0);
            break;
        case SmokeColor::RED:
            setColor(1.0, 0.0, 0.0);
            break;
        case SmokeColor::GREEN:
            setColor(0.0, 1.0, 0.0);
            break;
        case SmokeColor::BLUE:
            setColor(0.0, 0.0, 1.0);
            break;
    }
}
    
void FluidSim::updateSimulation() {
    float timestep = updateTimestep();

    if (gpu) {
        gpu_solver::update(this);
    } else {
        cpu_solver::update(this);
    }
}

void FluidSim::addDensity(int x, int y) {
    for (int i = -smokeSize+1; i < smokeSize+1; i++) {
        for (int j = -smokeSize+1; j < smokeSize+1; j++) {
            int idx = (y + i) * width + x + j;
            if (idx < 0 || idx >= width * height)
                continue;

            tempAdded[idx] += -1 * kd * tempDelta;

            for (int k = 0; k < 3; k++) {
                denseAdded[k][idx] = currColor[k];
                denseRGBA[idx*4+k] = 255*currColor[k];
            }
        }
    }
}

void FluidSim::addVelocity(int x, int y, float dx, float dy) {
    int idx = y * width + x;
    vxAdded[idx] += 0.5f * width * dx;
    vyAdded[idx] += 0.5f * height * -1*dy;
}

void FluidSim::addBoundary(int x, int y) {
    // first, just fill in the squares needed
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            int idx = (y + i) * width + x + j;
            if (idx < 0 || idx >= width * height)
                continue;

            setCenter(idx, bounds);

            for (int k = 0; k < 3; k++) {
                denseRGBA[idx*4+k] = 255*currColor[k];
            }
        }
    }  

    // then, update surrounding squares
    for (int i = -2; i < 3; i++) {
        for (int j = -2; j < 3; j++) {
            int idx = (y + i) * width + x + j;
            if (idx < 0 || idx >= width * height)
                continue;

            Boundary b = cellType(idx, bounds);
            setCellType(b, idx, bounds);
        }
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
