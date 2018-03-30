#include "julia.h"

__device__ cTYPE juliaFunctor(cTYPE p, cTYPE c)
{
    return cuCaddf(cuCmulf(p, p), c);
}

__device__ int evolveComplexPoint(cTYPE p, cTYPE c)
{
    int i = 1;
    while (i <= MAXITERATIONS && cuCabsf(p) <= 4) {
        p = juliaFunctor(p, c);
        i++;
    }
    return i;
}

__device__ cTYPE convertToComplex(int x, int y, float zoom)
{
    TYPE jx = 2 * (x / 1.0 - W / 2.0) / (0.5 * zoom * W) + moveX;
    TYPE jy = (y / 1.0 - H / 2.0) / (0.5 * zoom * H) + moveY;
    return cMakecuComplex(jx,jy);
}

__global__ void computeJulia(int *data, cTYPE c, float zoom)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < W && j < H){
        cTYPE p = convertToComplex(i, j, zoom);
        data[i * H + j] = evolveComplexPoint(p, c);
    }
}
