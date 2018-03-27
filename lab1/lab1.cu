#include "lab1.h"
static const unsigned W = 640;
static const unsigned H = 480;
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

__global__ void Draw(uint8_t *yuv, int t)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    int idx = i + j * W;

    /* Y */
    yuv[idx] = t * 255 / NFRAME;
    /* U */
    yuv[i / 2 + j * W / 2 + W * H] =  128;
    /* V */
    yuv[i + j * W / 4  + W * H + W * H / 4] =  128;
}

void Lab1VideoGenerator::Generate(uint8_t *yuv) {
    dim3 grid(W);
    dim3 block(H);
    Draw<<<grid, block>>>(yuv, impl->t);
	++(impl->t);
}
