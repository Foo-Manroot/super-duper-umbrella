GCC=gcc
NVCC=nvcc
CFLAGS=-Wall
OBJFLAGS=-Wall -c

.PHONY: clean all

all: cpu

cuda: joyas.cu
	$(NVCC) joyas.cu -o joyas-CUDA

cpu: joyas.o
	$(GCC) $(CFLAGS) joyas.o -o joyas

joyas.o: joyas.c joyas.h
	$(GCC) $(OBJFLAGS) joyas.c -o joyas.o

#### --------------------------------------------------- ####
# Compila con las opciones necesarias para perfilar con gprof
gprof: gprof.o
	$(GCC) $(CFLAGS) prof.o -o prof -pg

gprof.o: joyas.c joyas.h
	$(GCC) $(OBJFLAGS) mult_matrices.c -o prof.o -pg
#### --------------------------------------------------- ####

clean:
	rm -f *.o
	rm -f joyas
	rm -f joyas-CUDA
	rm -f prof
	rm -f gmon.out profile.txt
