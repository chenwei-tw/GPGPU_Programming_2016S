NVCC := /usr/local/cuda-9.1/bin/nvcc
CFLAGS := -std=c++11 -arch sm_50

main: lab3.cu pgm.cpp main.cu
	$(NVCC) $(CFLAGS) -I../utils -o $@ $^

output: main
	./main img_background.ppm img_target.ppm img_mask.pgm 600 130 img_output.ppm
	convert img_output.ppm img_output.jpg
