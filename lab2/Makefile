NVCC := /usr/local/cuda-9.1/bin/nvcc
CFLAGS := -std=c++11 -arch sm_50

main: counting.cu main.cu
	$(NVCC) $(CFLAGS) -I../utils -o $@ $^
