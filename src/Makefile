GCC=gcc
NVCC=nvcc
CFLAGS=-Wall
OBJFLAGS=-Wall -c
LIB=ar rcs


####
# Ayuda para el Makefile:
# 	-> $@ significa "el objetivo"
# 	-> $^ significa "los requisitos"
# 	-> $< significa "sólo el primer requisito"
####

.PHONY: clean all

all: candy

####
#cuda: joyas.cu
#	$(NVCC) joyas.cu -o joyas-CUDA
#----

####
candy: candy.o libparser.a
	$(GCC) $(CFLAGS) $^ -o $@

candy.o: candy.c candy.h common.h
	$(GCC) $(OBJFLAGS) $< -o $@
#----

####
# Biblioteca para el intérprete de comandos
libparser.a: parser.o
	$(LIB) $@ $^

parser.o: parser.c libparser.h common.h
	$(GCC) $(OBJFLAGS) $< -o $@
#----

#### --------------------------------------------------- ####
# Compila con las opciones necesarias para perfilar con gprof
gprof: gprof.o
	$(GCC) $(CFLAGS) $^ -o $@ -pg

gprof.o: joyas.c joyas.h
	$(GCC) $(OBJFLAGS) $< -o $@ -pg
#### --------------------------------------------------- ####

clean:
	rm -f *.o *.a
	rm -f candy
	rm -f joyas-CUDA
	rm -f gprof
	rm -f gmon.out profile.txt
