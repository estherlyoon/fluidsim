#include "gpu_solver.cuh"
#include "common.cuh"

#include <cstring>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s: %d,", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)): \
        exit(-1); \
    } \
}

namespace gpu_solver {

// TODO do for only boundary pixels? or id == 0?
__device__ void d_setBoundary(float* v, int w, int h, int bcond) {
    /*
    for (int i = 1; i < h; i++) {
        // vertical boundaries
        v[UV(0,i,w)] = bcond == CONTAINED_X ? -v[UV(1,i,w)] : v[UV(1,i,w)];
        v[UV(w-1,i,w)] = bcond == CONTAINED_X ? -v[UV(w-2,i,w)] : v[UV(w-2,i,w)];
    }

    for (int i = 1; i < w; i++) {
        // horizontal boundaries
        v[UV(i,0,w)] = bcond == CONTAINED_Y ? -v[UV(i,1,w)] : v[UV(i,1,w)];
        v[UV(i,h-1,w)] = bcond == CONTAINED_Y ? -v[UV(i,h-2,w)] : v[i,h-2,w];
    }

    v[UV(0,0,w)] = 0.5f * (v[UV(1,0,w)] + v[UV(0,1,w)]);
    v[UV(0,h-1,w)] = 0.5f * (v[UV(1,h-1,w)] + v[UV(0,h-2,w)]);
    v[UV(w-1,0,w)] = 0.5f * (v[UV(w-2,0,w)] + v[UV(w-1,1,w)]);
    v[UV(w-1,h-1,w)] = 0.5f * (v[UV(w-2,h-1,w)] + v[UV(w-1,h-2,w)]);
    */
}

__global__ void d_advect(float* vx, float* vy, float* quantity, float* tmpQuantity, float timestep, unsigned int w, unsigned int h, int bcond) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= (w-1) || y >= (h-1))
        return;

    // trace current cell backwards
    int idx = UV(x, y, w);
    float xPos = (float)x - timestep * vx[idx];
    float yPos = (float)y - timestep * vy[idx];

    // ignore OOB positions
    if (xPos < 0.5f || xPos > w-1.5f || yPos < 0.5f || yPos > h-1.5f)
        return;

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
    quantity[idx] = b1*(a1*tmpQuantity[UV(x0,y0,w)] + a0*tmpQuantity[UV(x1,y0,w)])
                + b0*(a1*tmpQuantity[UV(x0,y1,w)] + a0*tmpQuantity[UV(x1,y1,w)]);
    
    // SYNC

    //setBoundary(quantity, w, h, bcond);
}

__global__ void d_addSources(float* src, float* dest, int w, int h, float ts, float clamp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * w + x;

    if (x >= w || y >= h)
        return;

    if (clamp)
        dest[i] = fminf(clamp, dest[i] + ts*src[i]);
    else
        dest[i] += ts*src[i];
    src[i] = 0;
}

__global__ void d_diffuse(float* densities, float* tmpDensities, float diff_rate, int w, int h, float ts, int iters, int bcond) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= (w-1) || y >= (h-1))
        return;

    float a = ts * diff_rate * w * h;

    for (int k = 0; k < iters; k++) {
        densities[UV(x,y,w)] = (tmpDensities[UV(x,y,w)] +
                        a * (tmpDensities[UV(x,y-1,w)] + tmpDensities[UV(x,y+1,w)]
                             + tmpDensities[UV(x-1,y,w)] + tmpDensities[UV(x+1,y,w)]))
                            / (1 + 4 * a);
        
        //setBoundary(densities, w, h, bcond);
    }
}

__global__ void d_project(float* vx, float* vy, float* p, float* div, int w, int h, int iters) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= (w-1) || y >= (h-1))
        return;

    float h0 = 1.0f / h;
    float w0 = 1.0f / w;

    div[UV(x,y,w)] = -0.5f*h0*(vy[UV(x,y+1,w)] - vy[UV(x,y-1,w)]
                         + vx[UV(x+1,y,w)] - vx[UV(x-1,y,w)]);
    p[UV(x,y,w)] = 0;

    // SYNC

    //setBoundary(p, w, h, CONTINUOUS);
    //setBoundary(div, w, h, CONTINUOUS);

    // SYNC

    // set bnd for div, p
    for (int i = 0; i < iters; i++) {
        p[UV(x,y,w)] = (div[UV(x,y,w)] + p[UV(x-1,y,w)] + p[UV(x+1,y,w)]
                        + p[UV(x,y-1,w)] + p[UV(x,y+1,w)]) / 4.0f;
        // SYNC
        //setBoundary(p, w, h, CONTINUOUS);
        // SYNC
    }

    vx[UV(x,y,w)] -= 0.5 * (p[UV(x+1,y,w)] - p[UV(x-1,y,w)]) / w0;
    vy[UV(x,y,w)] -= 0.5 * (p[UV(x,y+1,w)] - p[UV(x,y-1,w)]) / h0;

    // SYNC
    //setBoundary(vx, w, h, CONTAINED_X);
    //setBoundary(vy, w, h, CONTAINED_Y);
}

__global__ void d_updateColors(float* densities, uint8_t* RGBA, uint8_t* res, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
        return;

    float density = densities[UV(x,y,w)];
    for (int c = 0; c < 3; c++) {
        res[UV(x,y,w)*4+c] = density * RGBA[UV(x,y,w)*4+c];
    }
}


void update(FluidSim* sim) {
    // todo:
    // copy density and velocity sources to cuda pointers
    // replace current names with cuda pointers (actually just allocate)
    // sync after every call

    // derive kernel configuration
    // number of threads in each direction (TODO get over 1024)
    dim3 blockSize(sim->width, sim->height);
    int bx = (sim->width + blockSize.x - 1)/blockSize.x ;
    int by = (sim->height + blockSize.y - 1)/blockSize.y ;
    dim3 gridSize = dim3(bx, by);

    // solve velocity
    d_addSources<<<gridSize, blockSize>>>(sim->vxAdded, sim->vx, sim->width, sim->height, 1.0f, 0.0f);
    d_addSources<<<gridSize, blockSize>>>(sim->vyAdded, sim->vy, sim->width, sim->height, 1.0f, 0.0f);

    float visc = 0.5;
    swap(&sim->vx, &sim->tmpV);
    d_diffuse<<<gridSize, blockSize>>>(sim->vx, sim->tmpV, visc, sim->width, sim->height, 1.0, 20, CONTAINED_X);
    swap(&sim->vy, &sim->tmpV);
    d_diffuse<<<gridSize, blockSize>>>(sim->vy, sim->tmpV, visc, sim->width, sim->height, 1.0, 20, CONTAINED_Y);

    d_project<<<gridSize, blockSize>>>(sim->vx, sim->vy, sim->tmpV, sim->tmpU, sim->width, sim->height, 40);

    swap(&sim->vx, &sim->tmpV);
    d_advect<<<gridSize, blockSize>>>(sim->vx, sim->vy, sim->vx, sim->tmpV, 1.0, sim->width, sim->height, CONTAINED_X);
    swap(&sim->vy, &sim->tmpV);
    d_advect<<<gridSize, blockSize>>>(sim->vx, sim->vy, sim->vy, sim->tmpV, 1.0, sim->width, sim->height, CONTAINED_Y);

    d_project<<<gridSize, blockSize>>>(sim->vx, sim->vy, sim->tmpV, sim->tmpU, sim->width, sim->height, 40);
     
    // solve density
    float diff_rate = 0.8;
    d_addSources<<<gridSize, blockSize>>>(sim->denseAdded, sim->densities, sim->width, sim->height, 1.0, 1.0f);

    swap(&sim->densities, &sim->tmpV);
    d_diffuse<<<gridSize, blockSize>>>(sim->densities, sim->tmpV, diff_rate, sim->width, sim->height, 1.0, 20, CONTINUOUS);
    swap(&sim->densities, &sim->tmpV);
    d_advect<<<gridSize, blockSize>>>(sim->vx, sim->vy, sim->densities, sim->tmpV, 1.0, sim->width, sim->height, CONTINUOUS);
    d_updateColors<<<gridSize, blockSize>>>(sim->densities, sim->RGBA, sim->denseRGBA, sim->width, sim->height);

}


}
