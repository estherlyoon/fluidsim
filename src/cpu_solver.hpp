#ifndef __CPU_SOLVER_HPP__
#define __CPU_SOLVER_HPP__

#include "fluidsim.hpp"

namespace cpu_solver {

    void update(FluidSim* sim, float timestep);
    void advect(float* vx, float* vy, float* quantity, float* tmpQuantity, float timestep, unsigned int w, unsigned int h, int bcond);
    void diffuse(float* densities, float* tmpDensities, float diff_rate, int w, int h, float ts, int iters, bool bcond);

}


#endif
