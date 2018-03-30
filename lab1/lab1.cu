#include "lab1.h"
#include "julia.h"
#include <cuComplex.h>
#include <string.h>

static const unsigned NFRAME = 240;

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

__device__ void dwell_color(int *r, int *g, int *b, int dwell)
{
    if(dwell >= MAX_DWELL) {
        *r = *g = *b = 128;
    } else {
        if(dwell < 0)
            dwell = 0;
        if(dwell <= CUT_DWELL) {
            *b = *g = 0;
            *r = 128 + dwell * 127 / (CUT_DWELL);
        } else {
            *r = 255;
            *b = *g = (dwell - CUT_DWELL) * 255 / (MAX_DWELL - CUT_DWELL);
        }
    }
}

__global__ void Draw(uint8_t *yuv, int *src)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    int idx = j + i * W;
    int r, g, b;

    dwell_color(&r, &g, &b, src[idx]);

    /* Y */
    yuv[idx] = r;

    if (i % 2 == 0 && j % 2 == 0) {
        /* U */
        yuv[W * H + i / 2 + (j / 2) * (W / 2)] = (r * (-0.169) - 0.331 * g + 0.5 * b) + 128.0;
        /* V */
        yuv[W * H + W * H / 4 + i / 2 + (j / 2) * (W / 2)] = 0.5 * r - 0.419 * g - 0.081 * b + 128.0;
    }
}

int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

void Lab1VideoGenerator::Generate(uint8_t *yuv) {
    /* Compute Graph of Julia Set */
    dim3 bs(BLOCKX, BLOCKY);
	dim3 gs(divup(W, bs.x), divup(H, bs.y));
    /* Draw */
    dim3 grid(H);
    dim3 block(W);
    /* Other arguments */
    int i = impl->t;
    cTYPE c0;

	size_t dataSize= sizeof(int) * W * H;
	int* hdata;
	CUDA_CHECK_RETURN(cudaMallocHost(&hdata, sizeof(int) * W * H));
	int* ddata;
	CUDA_CHECK_RETURN(cudaMalloc(&ddata, sizeof(int) * W * H));

    /* Compute graph */
    c0 = cMakecuComplex(STARTRE + i * INCRE, STARTI + i * INCI);
    computeJulia<<<gs,bs>>>(ddata, c0, ZOOM);
    
    /* Draw graph */
    Draw<<<grid, block>>>(yuv, ddata);
    
    /* Free */
    cudaFree(ddata);
    cudaFree(hdata);

    ++(impl->t);
}
