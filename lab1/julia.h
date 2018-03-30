#ifndef JULIA_H_
#define JULIA_H_

#include <cuComplex.h>

#define TYPE float
#define cTYPE cuFloatComplex
#define cMakecuComplex(re,i) make_cuFloatComplex(re,i)

#define MAXITERATIONS 256
#define MAX_DWELL MAXITERATIONS
#define CUT_DWELL MAX_DWELL / 4
#define W 1024
#define H 1024
#define BLOCKX 32
#define BLOCKY 32
#define moveX 0
#define moveY 0
#define INCRE 0.00000003
#define INCI -0.00009
#define STARTRE -0.75
#define STARTI 0.09
#define ZOOM 3.2

#define CUDA_CHECK_RETURN(value) {											\
		cudaError_t _m_cudaStat = value;										\
		if (_m_cudaStat != cudaSuccess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
					exit(1);															\
		} }

__device__ static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex x)
{
	double factor = exp(x.x);
	return make_cuDoubleComplex(factor * cos(x.y), factor * sin(x.y));
}

__device__ static __inline__ cuFloatComplex cuCexp(cuFloatComplex x)
{
	float factor = exp(x.x);
	return make_cuFloatComplex(factor * cos(x.y), factor * sin(x.y));
}

__device__ cTYPE juliaFunctor(cTYPE p, cTYPE c);

__device__ int evolveComplexPoint(cTYPE p, cTYPE c);

__device__ cTYPE convertToComplex(int x, int y, float zoom);

__global__ void computeJulia(int *data, cTYPE c, float zoom);

#endif
