# 2D Smoke Simulation in CUDA
Two-dimensional smoke simulation written in C++ and CUDA. I used the Navier-Stokes equation and the methods described in Stam's 2003 paper "Real-Time Fluid Dynamics for Games" for my implementation. 

https://user-images.githubusercontent.com/43528786/181171961-d3cc7ae3-b453-499b-90b1-87175d612f0a.mp4

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
