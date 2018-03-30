NVCC := /usr/local/cuda-9.1/bin/nvcc
CFLAGS := -std=c++11

all: video

julia.o: julia.cu
	$(NVCC) $(CFLAGS) -c -o $@ $<

main: main.cu lab1.cu julia.o
	 $(NVCC) $(CFLAGS) -I../utils -o $@ $^

video: main
	./main
	avconv -i result.y4m result.mkv
