NVCC := /usr/local/cuda-9.1/bin/nvcc
CFLAGS := -std=c++11

all: video

main: main.cu lab1.cu
	 $(NVCC) $(CFLAGS) -I../utils -o $@ $^

video: main
	./main
	avconv -i result.y4m result.mkv
