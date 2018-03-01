NVCC := /usr/local/cuda-9.1/bin/nvcc

main: main.cu
	$(NVCC) -std=c++11 -I../utils -o main main.cu

clean:
	rm main
