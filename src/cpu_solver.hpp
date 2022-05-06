#ifndef __CPU_SOLVER_HPP__
#define __CPU_SOLVER_HPP__

#include "fluidsim.hpp"

namespace cpu_solver {

    void update(FluidSim* sim, float timestep);
    void advect(float* vx, float* vy, uint8_t* RGBA, float timestep, unsigned int w, unsigned int h);

};

#endif
