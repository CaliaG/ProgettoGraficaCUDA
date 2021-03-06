CUDA_PATH?=/usr/local/cuda
HOST_COMPILER=g++
NVCC=$(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVPROF_FLAGS=--metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer

# debug vs release
NVCCFLAGS_DEBUG= -g -G
#NVCCFLAGS=
# > Tesla K80
NVCCFLAGS=-arch=sm_37

all: clean main.o compile run
debug: clean dbg

compile: main.o
	$(NVCC) $(NVCCFLAGS) -o main main.o

main.o:
	$(NVCC) $(NVCCFLAGS) -o main.o -c main.cu

dbg:
	$(NVCC) $(NVCCFLAGS_DEBUG) -o main.o -c main.cu
	$(NVCC) $(NVCCFLAGS_DEBUG) -o main main.o

run: main
	rm -f image.jpg
	./main

profile: main
	nvprof ./main

profile_metrics: main
	nvprof $(NVPROF_FLAGS) ./main

clean:
	rm -f main main.o image.jpg image.ppm