#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

void CountPosition1(const char *text, int *pos, int text_size)
{
    char *linebreak;
    /* Create an array contains linebreak */
    cudaMalloc((void **) &linebreak, text_size);
    cudaMemset(linebreak, '\n', text_size);

    /* Create an array from comparing linebreak and original text */
    thrust::transform(thrust::device, text, text + text_size, linebreak, pos, thrust::not_equal_to<const char>());
    /* Prefix Sum */
    thrust::inclusive_scan_by_key(thrust::device, pos, pos + text_size, pos, pos);

    cudaFree(linebreak);
}

#define BLOCK_SIZE 32

__global__ void mytransform(const char *text, int *pos, int text_size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    int idx = i * BLOCK_SIZE + j;

    if (idx < text_size) {
        pos[idx] = 1;
        if (text[idx] == '\n')
            pos[idx] = 0;
    }
}

__global__ void myprefixsum(int *pos, int text_size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    int idx = i * BLOCK_SIZE + j;

    if (idx < text_size - 1) {
        if (pos[idx] == 0 && pos[idx + 1] == 1) {
            ++idx;
            int acc = 1;
            do {
                pos[idx] = acc;
                ++acc;
                ++idx;
            } while (idx < text_size && pos[idx] == 1);
        }
    }

    if (idx == 0 && pos[idx] == 1) {
        int acc = 1;
        do {
            pos[idx] = acc;
            ++acc;
            ++idx;
        } while (pos[idx] == 1);
    }
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    mytransform<<<(text_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(text, pos, text_size);
    myprefixsum<<<(text_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(pos, text_size);
}
