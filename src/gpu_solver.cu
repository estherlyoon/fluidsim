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
    printf("Error: %s:%d, ", __FILE__, __LINE__); \
    printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
    exit(1); \
   } \
}

namespace gpu_solver {

__device__ void d_setBoundary(float* v, int x, int y, int w, int h, int bcond) {
    // vertical boundaries
    if (x == 1) {
        v[UV(0,y,w)] = bcond == CONTAINED_X ? -v[UV(1,y,w)] : v[UV(1,y,w)];
    } else if (x == w-2) {
        v[UV(w-1,y,w)] = bcond == CONTAINED_X ? -v[UV(w-2,y,w)] : v[UV(w-2,y,w)];
    // horizontal boundaries
    } else if (y == 1) {
        v[UV(x,0,w)] = bcond == CONTAINED_Y ? -v[UV(x,1,w)] : v[UV(x,1,w)];
    } else if (y == h-2) {
        v[UV(x,h-1,w)] = bcond == CONTAINED_Y ? -v[UV(x,h-2,w)] : v[UV(x,h-2,w)];
    }

    __syncthreads();

    if (x == 1 && y == 1)
        v[UV(0,0,w)] = 0.5f * (v[UV(1,0,w)] + v[UV(0,1,w)]);
    if (x == 1 && y == h-2)
        v[UV(0,h-1,w)] = 0.5f * (v[UV(1,h-1,w)] + v[UV(0,h-2,w)]);
    if (x == w-2 && y == 1)
        v[UV(w-1,0,w)] = 0.5f * (v[UV(w-2,0,w)] + v[UV(w-1,1,w)]);
    if (x == w-2 && y == h-2)
        v[UV(w-1,h-1,w)] = 0.5f * (v[UV(w-2,h-1,w)] + v[UV(w-1,h-2,w)]);

    __syncthreads();
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
    
    d_setBoundary(quantity, x, y, w, h, bcond);
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
        
        d_setBoundary(densities, x, y, w, h, bcond);
        __syncthreads();
    }
}

__global__ void d_project1(float* vx, float* vy, float* p, float* div, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= (w-1) || y >= (h-1))
        return;

    float h0 = 1.0f / h;

    div[UV(x,y,w)] = -0.5f*h0*(vy[UV(x,y+1,w)] - vy[UV(x,y-1,w)]
                         + vx[UV(x+1,y,w)] - vx[UV(x-1,y,w)]);
    p[UV(x,y,w)] = 0;

    d_setBoundary(p, x, y, w, h, CONTINUOUS);
    d_setBoundary(div, x, y, w, h, CONTINUOUS);
}

__global__ void d_project2(float* vx, float* vy, float* p, float* div, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= (w-1) || y >= (h-1))
        return;

    p[UV(x,y,w)] = (div[UV(x,y,w)] + p[UV(x-1,y,w)] + p[UV(x+1,y,w)]
                    + p[UV(x,y-1,w)] + p[UV(x,y+1,w)]) / 4.0f;
    d_setBoundary(p, x, y, w, h, CONTINUOUS);
}

__global__ void d_project3(float* vx, float* vy, float* p, float* div, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= (w-1) || y >= (h-1))
        return;

    float h0 = 1.0f / h;
    float w0 = 1.0f / w;

    vx[UV(x,y,w)] -= 0.5 * (p[UV(x+1,y,w)] - p[UV(x-1,y,w)]) / w0;
    vy[UV(x,y,w)] -= 0.5 * (p[UV(x,y+1,w)] - p[UV(x,y-1,w)]) / h0;

    d_setBoundary(vx, x, y, w, h, CONTAINED_X);
    d_setBoundary(vy, x, y, w, h, CONTAINED_Y);
}

__global__ void d_updateColors(float* densities, uint8_t* res, int w, int h, int color) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
        return;

    float density = densities[UV(x,y,w)];

    res[UV(x,y,w)*4+color] = density * 255.0f;
    res[UV(x,y,w)*4+3] = 255;
}

void projectAll(dim3 gridSize, dim3 blockSize, float* vx, float* vy, float* p, float* div, int w, int h, int iters) {
    d_project1<<<gridSize, blockSize>>>(vx, vy, p, div, w, h);
    for (int i = 0; i < iters; i++)
        d_project2<<<gridSize, blockSize>>>(vx, vy, p, div, w, h);
    d_project3<<<gridSize, blockSize>>>(vx, vy, p, div, w, h);
}

void solveDensity(FluidSim* sim, int c, dim3 gridSize, dim3 blockSize) {
    float diff_rate = 0.8;
    d_addSources<<<gridSize, blockSize>>>(sim->cudaDenseAdded[c], sim->densities[c], sim->width, sim->height, sim->timeDelta, 1.0f);

    swap(&sim->densities[c], &sim->tmpV);
    d_diffuse<<<gridSize, blockSize>>>(sim->densities[c], sim->tmpV, diff_rate, sim->width, sim->height, sim->timeDelta, 20, CONTINUOUS);
    swap(&sim->densities[c], &sim->tmpV);
    d_advect<<<gridSize, blockSize>>>(sim->vx, sim->vy, sim->densities[c], sim->tmpV, sim->timeDelta, sim->width, sim->height, CONTINUOUS);
    d_updateColors<<<gridSize, blockSize>>>(sim->densities[c], sim->cudaDenseRGBA, sim->width, sim->height, c);
 
}


void update(FluidSim* sim) {
    // copy density and velocity sources to cuda pointers
    cudaMemcpy(sim->cudaVxAdded, sim->vxAdded, sizeof(float)*sim->width*sim->height, cudaMemcpyHostToDevice);
    cudaMemcpy(sim->cudaVyAdded, sim->vyAdded, sizeof(float)*sim->width*sim->height, cudaMemcpyHostToDevice);
    cudaMemcpy(sim->cudaDenseAdded[0], sim->denseAdded[0], sizeof(float)*sim->width*sim->height, cudaMemcpyHostToDevice);
    cudaMemcpy(sim->cudaDenseAdded[1], sim->denseAdded[1], sizeof(float)*sim->width*sim->height, cudaMemcpyHostToDevice);
    cudaMemcpy(sim->cudaDenseAdded[2], sim->denseAdded[2], sizeof(float)*sim->width*sim->height, cudaMemcpyHostToDevice);
    CHECK( cudaPeekAtLastError()  );
    CHECK( cudaDeviceSynchronize()  );
    cudaMemcpy(sim->cudaTempAdded, sim->tempAdded, sizeof(float)*sim->width*sim->height, cudaMemcpyHostToDevice);
    CHECK( cudaPeekAtLastError()  );
    CHECK( cudaDeviceSynchronize()  );

    memset(sim->vxAdded, 0, sizeof(float)*sim->width*sim->height);
    memset(sim->vyAdded, 0, sizeof(float)*sim->width*sim->height);
    memset(sim->tempAdded, 0, sizeof(float)*sim->width*sim->height);
    memset(sim->denseAdded[0], 0, sizeof(float)*sim->width*sim->height);
    memset(sim->denseAdded[1], 0, sizeof(float)*sim->width*sim->height);
    memset(sim->denseAdded[2], 0, sizeof(float)*sim->width*sim->height);
    CHECK( cudaPeekAtLastError()  );
    CHECK( cudaDeviceSynchronize()  );

    // derive kernel configuration
    const dim3 blockSize(32, 32);
    int bx = (sim->width + blockSize.x - 1)/blockSize.x;
    int by = (sim->height + blockSize.y - 1)/blockSize.y;
    const dim3 gridSize = dim3(bx, by);

    // solve velocity
    d_addSources<<<gridSize, blockSize>>>(sim->cudaVxAdded, sim->vx, sim->width, sim->height, sim->timeDelta, 0.0f);
    d_addSources<<<gridSize, blockSize>>>(sim->cudaVyAdded, sim->vy, sim->width, sim->height, sim->timeDelta, 0.0f);
    d_addSources<<<gridSize, blockSize>>>(sim->cudaTempAdded, sim->vy, sim->width, sim->height, sim->timeDelta, 0.0f);

    swap(&sim->vx, &sim->tmpV);
    d_diffuse<<<gridSize, blockSize>>>(sim->vx, sim->tmpV, sim->viscosity, sim->width, sim->height, sim->timeDelta, 20, CONTAINED_X);
    swap(&sim->vy, &sim->tmpV);
    d_diffuse<<<gridSize, blockSize>>>(sim->vy, sim->tmpV, sim->viscosity, sim->width, sim->height, sim->timeDelta, 20, CONTAINED_Y);

    projectAll(gridSize, blockSize, sim->vx, sim->vy, sim->tmpV, sim->tmpU, sim->width, sim->height, 40);

    swap(&sim->vx, &sim->tmpV);
    d_advect<<<gridSize, blockSize>>>(sim->vx, sim->vy, sim->vx, sim->tmpV, sim->timeDelta, sim->width, sim->height, CONTAINED_X);
    swap(&sim->vy, &sim->tmpV);
    d_advect<<<gridSize, blockSize>>>(sim->vx, sim->vy, sim->vy, sim->tmpV, sim->timeDelta, sim->width, sim->height, CONTAINED_Y);

    projectAll(gridSize, blockSize, sim->vx, sim->vy, sim->tmpV, sim->tmpU, sim->width, sim->height, 40);

    solveDensity(sim, 0, gridSize, blockSize);
    solveDensity(sim, 1, gridSize, blockSize);
    solveDensity(sim, 2, gridSize, blockSize);
     
    // copy RGBA values back to host
    cudaMemcpy(sim->denseRGBA, sim->cudaDenseRGBA, sizeof(uint8_t)*sim->width*sim->height*4, cudaMemcpyDeviceToHost);

}


}
