#include "common.cuh"

#include <sstream>
#include <iomanip>


__device__ __host__ Boundary cellType(int idx, uint8_t* bounds) {
    bool n = isNorth(idx, bounds);
    bool e = isEast(idx, bounds);
    bool w = isWest(idx, bounds);
    bool s = isSouth(idx, bounds);
    bool c = isCenter(idx, bounds);

    if (!c) {
        return FLUID;
    } else {
        if (n && !w && !s)
            return NE;
        else if (n && !e && w && !s)
            return NW;
        else if (!n && !e && w && s)
            return SW;
        else if (!n && e && !w && s)
            return SE;
        else if (n && !e && !w && !s)
            return NORTH;
        else if (!n && e && !w && !s)
            return EAST;
        else if (!n && !e && !w && s)
            return SOUTH;
        else if (!n && !e && w && !s)
            return WEST;
        else
            return NONE;
    }
}                                                

void setCellType(Boundary b, int idx, uint8_t* bounds) {
    switch(b) {
        case NORTH:
            setNorth(idx, bounds);
            break;
        case SOUTH:
            setSouth(idx, bounds);
            break;
        case EAST:
            setEast(idx, bounds);
            break;
        case WEST:
            setWest(idx, bounds);
            break;
        case NE:
            setNorth(idx, bounds);
            setEast(idx, bounds);
            break;
        case NW:
            setNorth(idx, bounds);
            setWest(idx, bounds);
            break;
        case SW:
            setSouth(idx, bounds);
            setWest(idx, bounds);
            break;
        case SE:
            setSouth(idx, bounds);
            setEast(idx, bounds);
            break;
        case CENTER:
            setCenter(idx, bounds);
            break;
        default:
            break;
    }
}

void swap(float** a, float** b) {
    float* tmp = *a;
    *a = *b;
    *b = tmp;
}

std::string ftos(float v) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << v;
    return stream.str();
}

