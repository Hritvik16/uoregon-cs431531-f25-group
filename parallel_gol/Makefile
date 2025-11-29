CC      = /usr/bin/gcc
CFLAGS  = -Wall -g -std=c99
CFLAGS += -O3 -fopenmp
CFLAGS += -D_POSIX_C_SOURCE=200112L
LIBS    = -lm

NVCC      = nvcc
NVCCFLAGS = -O3

all: gol_serial gol_omp gol_cuda

gol_serial: gol_serial.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS) $(CUDA_LDFLAGS)

gol_omp: gol_omp.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS) $(CUDA_LDFLAGS)

gol_cuda: gol_cuda.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) $(INC) $(NVCCINC) -c $< -o $@ $(LIBS)

clean:
	rm -f *.o *.d *~ *.a *.so *.s gol_serial gol_omp gol_cuda
