# 2D Smoke Simulation in CUDA
Two-dimensional smoke simulation written in C++ and CUDA. I used the Navier-Stokes equation and the methods described in Stam's 2003 paper "Real-Time Fluid Dynamics for Games" for my implementation. 

![Video Demo](https://drive.google.com/file/d/1_fVs2Sfob9b669ooV9sL_UCzuXitozdX/view?resourcekey)

## Dependencies
- SFML > 2.0
- Boost
- C++
- NVIDIA CUDA Toolkit

## To Run
- `mkdir build && cd build`
- `cmake ..`
- in the build directory, `make`
- `./fluid-sim`

## Command-line Options
- `--gpu <bool>` to run the CUDA version
- `--width <int>` to specify simulation width in pixels
- `--height <int>` to specify simulation height in pixels

## Controls
- Press R key to reset the simulation
- Press Space key to start or pause the simulation
- Left-click and drag to add dye/draw boundaries
- Right-click and drag to add force
- Press B key to go in and out of boundary-drawing mode

Press number keys for different smoke/boundary colors:
- 1: White
- 2: Red
- 3: Green
- 4: Blue

Note: the arbitrary edge detection feauture is slightly bugged-- I couldn't figure out how to properly handle density values at boundaries so the smoke that isn't advected away with the velocity field just kind of disappears into the boundary. I also didn't have time to implement arbitrary edge detection in CUDA (did an hour before this was due), so it's only available for the CPU option (for now!).

Sorry if the CUDA version is buggy... the latency using X11 was so bad that it was difficult to test.
