#include "lab3.h"
#include <cstdio>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define ITER 16000

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void initializeJacobi(
        const float *mask,
        const float *background,
        const float *target,
        float *count,
        float *partial_sum,
        float *result,
        const int ht, const int wt,
        const int oy, const int ox,
        const int wb, const int hb,
        int channel
        )
{
    int yt = blockIdx.y * blockDim.y + threadIdx.y;
    int xt = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_t = wt * yt + xt;
    int yoffset[] = {0, -1, 1, 0};
    int xoffset[] = {-1, 0, 0, 1};

    if (yt >= 0 && yt < ht && xt >= 0 && wt > xt) {
        if (mask[pos_t] > 127.0f) {
            result[pos_t] = target[pos_t * 3 + channel];
            for (int i = 0; i < 4; i++) {
                int ycorner = yt + yoffset[i];
                int xcorner = xt + xoffset[i];
                int corner = ycorner * wt + xcorner;
                int ycorner_t = yt + oy + yoffset[i];
                int xcorner_t = xt + ox + xoffset[i];

                if (ycorner_t >= 0 && ycorner_t < hb && xcorner_t >= 0 && xcorner_t < wb) { 
                    count[pos_t]++;
                    if (ycorner >= 0 && ycorner < ht && \
                            xcorner >= 0 && xcorner < wt) {
                        partial_sum[pos_t] += target[pos_t * 3 + channel] - \
                                              target[corner * 3 + channel];
                    }
                }
            }
        } else {
            result[pos_t] = partial_sum[pos_t] = background[((yt + oy) * wb + xt + ox) * 3 + channel];
        }
    }
}

__global__ void Jacobi(
        float *result,
        float *count,
        float *partial_sum,
        const float *background,
        const float *mask,
        const int ht, const int wt,
        const int wb, const int hb,
        const int oy, const int ox,
        int channel
        )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int yt = idx / wt, xt = idx % wt;
    int yoffset[] = {0, -1, 1, 0};
    int xoffset[] = {-1, 0, 0, 1};
    float sum = 0;

    if (idx < ht * wt && mask[idx] > 127.0f) {
        for (int i = 0; i < 4; i++) {
            int ycorner = yt + yoffset[i];
            int xcorner = xt + xoffset[i];
            int corner = ycorner * wt + xcorner;
            int ycorner_t = yt + oy + yoffset[i];
            int xcorner_t = xt + ox + xoffset[i];

            if (ycorner_t >= 0 && ycorner_t < hb && xcorner_t >= 0 && xcorner_t < wb) {
                if (ycorner >= 0 && ycorner < ht && \
                        xcorner >= 0 && xcorner < wt) {
                    sum -= result[corner];
                } else {
                    sum -= background[(ycorner_t * wb + xcorner_t) * 3 + channel]; 
                }
            }
        }
        result[idx] = (partial_sum[idx] - sum) / count[idx];
    }
}

__global__ void attach(
        float *output,
        float *result,
        const int wt, const int ht,
        const int oy, const int ox,
        const int wb, const int hb,
        int channel
        )
{
    int yt = blockIdx.y * blockDim.y + threadIdx.y;
    int xt = blockIdx.x * blockDim.x + threadIdx.x;
    int yb = oy + yt, xb = ox + xt;
    int curt = wt * yt + xt;
    int curb = wb * yb + xb;

    if (yt < ht && xt < wt)
        if (yb >= 0 && yb < hb && xb >= 0 && xb < wb)
            output[curb * 3 + channel] = result[curt];
}

void PoissonImageCloning(
        const float *background,
        const float *target,
        const float *mask,
        float *output,
        const int wb, const int hb, const int wt, const int ht,
        const int oy, const int ox
        )
{
    float *count;
    float *partial_sum;
    float *result;

    cudaMalloc((void **) &count, wt * ht * sizeof(float));
    cudaMalloc((void **) &partial_sum, wt * ht * sizeof(float));
    cudaMalloc((void **) &result, wt * ht * sizeof(float));

    cudaMemcpy(output, background, 3 * wb * hb * sizeof(float), \
            cudaMemcpyDeviceToDevice);

    for(int channel = 0; channel < 3; channel++){
        cudaMemset((void *) count, 0, wt * ht * sizeof(float));
        cudaMemset((void *) partial_sum, 0, wt * ht * sizeof(float));
        cudaMemset((void *) result, 0, wt * ht * sizeof(float));
        initializeJacobi<<<dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16)>>>(mask, background, target, count, partial_sum, result, ht, wt, oy, ox, wb, hb, channel);

        for (int iter = 0; iter < ITER; iter++) {
            Jacobi<<<((ht * wt) / 32) + 1, 32>>>(result, count, partial_sum, background, mask, ht, wt, wb, hb, oy, ox, channel);
            attach<<<dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16)>>>(output, result, wt, ht, oy, ox, wb, hb, channel);
        }
    }
}
