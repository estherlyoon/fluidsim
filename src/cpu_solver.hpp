#ifndef __CPU_SOLVER_HPP__
#define __CPU_SOLVER_HPP__

#include "fluidsim.hpp"

namespace cpu_solver {

    void update(FluidSim* sim);
    void advect(float* vx, float* vy, float timestep, unsigned int w, unsigned int h);

};

#endif
