#include "fluidsim.hpp"
#include "cpu_solver.hpp"
#include "gpu_solver.cuh"

#include <SFML/Graphics.hpp>
#include <random>
#include <cuda_runtime_api.h>
#include <cuda.h>

FluidSim::FluidSim(unsigned int w, unsigned int h, bool gpu) : width(w), height(h), gpu(gpu), xPoint(0), yPoint(0) {
    vxAdded = new float[w * h]();
    vyAdded = new float[w * h]();
    denseAdded = new float[w * h]();
    denseRGBA = new uint8_t[w * h * 4]();
    RGBA = new uint8_t[w * h * 4]();

    if (gpu) allocDevice();
    else allocHost();
    
    std::random_device r;

    // Choose a random mean between 1 and 6
    std::default_random_engine e1(r());
    std::uniform_int_distribution<int> uniform_dist(1, w-1);
     
    // set opacity
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int i = y * w + x;
            int rand = uniform_dist(e1);
            // init base color to white
            for (int j = 0; j < 4; j++) 
                RGBA[i*4+j] = 255;
            if (x > 0 && y > 0 && x < w-1 && y < h-1 && (i > 12800 && i < 25600) || (i < 102400 && i > 80000)) {
                for (int j = 0; j < 3; j++) {
                    denseRGBA[i*4+j] = 255;
                }
                denseAdded[i] = 1.0;
                vyAdded[i] = 1.0;
                vxAdded[i] = 1.0;
            }
            denseRGBA[i*4+3] = 255;
        }
    }

    if (gpu)
        cudaMemcpy(cudaRGBA, RGBA, sizeof(uint8_t)*width*height*4, cudaMemcpyHostToDevice);

}

FluidSim::~FluidSim() {
 // TODO cuda free
}

void FluidSim::allocHost() {
    size_t dim = width * height;
    // assume fluid start with zero initial velocity and pressure
    vx = new float[dim]();
    vy = new float[dim]();
    pressures = new float[dim]();
    temperatures = new float[dim]();
    densities = new float[dim]();

    // point (x, y) on screen -> (y*width+x)*4
    tmpV = new float[dim];
    tmpU = new float[dim];
}

void FluidSim::allocDevice() {
    size_t dim = sizeof(float) * width * height;
    cudaMalloc((void**)&vx, dim);
    cudaMalloc((void**)&vy, dim);
    cudaMalloc((void**)&pressures, dim);
    cudaMalloc((void**)&temperatures, dim);
    cudaMalloc((void**)&densities, dim);
    cudaMalloc((void**)&tmpV, dim);
    cudaMalloc((void**)&tmpU, dim);
    cudaMalloc((void**)&cudaDenseAdded, dim);
    cudaMalloc((void**)&cudaVxAdded, dim);
    cudaMalloc((void**)&cudaVyAdded, dim);
    cudaMalloc((void**)&cudaRGBA, sizeof(uint8_t)*width*height*4);
    cudaMalloc((void**)&cudaDenseRGBA, sizeof(uint8_t)*width*height*4);

    // init to 0
    cudaMemset(vx, 0, dim);
    cudaMemset(vy, 0, dim);
    cudaMemset(tmpV, 0, dim);
    cudaMemset(tmpU, 0, dim);
    cudaMemset(densities, 0, dim);
    cudaMemset(cudaRGBA, 255, sizeof(uint8_t)*width*height);
    cudaMemset(cudaDenseRGBA, 255, sizeof(uint8_t)*width*height);
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
    printf("adding to x = %d, y = %d\n", x, y);
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
