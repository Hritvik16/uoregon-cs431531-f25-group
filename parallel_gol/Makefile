PRIMARY_PATH      = /usr/bin/mpiicc
SECONDARY_PATH    = mpiicc

ifneq ($(wildcard $(PRIMARY_PATH)),)
  CC = $(PRIMARY_PATH)
else
  CC = $(SECONDARY_PATH)
endif

CFLAGS  = -Wall -g -std=c99
CFLAGS += -O3 -fopenmp
CFLAGS += -D_POSIX_C_SOURCE=200112L
LIBS    = -lm

NVCC      = nvcc
NVCCFLAGS = -O3

all: gol_omp #gol_cuda

#gol_serial: gol_serial.o
#	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS) $(CUDA_LDFLAGS)

gol_omp: gol_omp.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS) $(CUDA_LDFLAGS)

#gol_cuda: gol_cuda.cu
#	$(NVCC) $(NVCCFLAGS) $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) $(INC) $(NVCCINC) -c $< -o $@ $(LIBS)

clean:
	-rm -f *.o *.d *~ *.a *.so *.s gol_serial gol_omp gol_cuda
	-del *.o *.d *~ *.a *.so *.s gol_serial gol_omp gol_cuda