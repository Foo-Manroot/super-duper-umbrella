# Para ver los registros usados por cada núcleo, compilar con
# NVCCFLAGS=--ptxas-options=-v

GCC=nvcc
NVCC=nvcc
CFLAGS=
NVCCFLAGS=
OBJFLAGS=-c
LIB=ar rcs

# Directorios
OUTDIR=obj
INCDIR=include
crear_dir=@mkdir -p $(@D)

####
# Ayuda para el Makefile:
# 	-> $@ significa "el objetivo"
# 	-> $^ significa "los requisitos"
# 	-> $< significa "sólo el primer requisito"
####

.PHONY: clean all

all: candy

####
# Objetivo principal
#
candy: $(OUTDIR)/cuda.o $(OUTDIR)/candy.o $(OUTDIR)/libutils.a
	$(GCC) $(CFLAGS) $^ -o $@
#----

####
# Archivo con los núcleos para CUDA
#
$(OUTDIR)/cuda.o: kernel.cu $(INCDIR)/kernel.cuh
	$(crear_dir)
	$(NVCC) $(OBJFLAGS) $(NVCCFLAGS) $< -o $@
#----

####
# Biblioteca para el intérprete de comandos y las funciones auxiliares
#
$(OUTDIR)/libutils.a: $(OUTDIR)/utils.o $(OUTDIR)/menu.o
	$(crear_dir)
	$(LIB) $@ $^

$(OUTDIR)/%.o: %.cpp $(INCDIR)/*.h
	$(crear_dir)
	$(GCC) $(OBJFLAGS) $< -o $@

#### --------------------------------------------------- ####
# Compila con las opciones necesarias para perfilar con gprof
#gprof: gprof.o
#	$(GCC) $(CFLAGS) $^ -o $@ -pg
#
#gprof.o: joyas.c joyas.h
#	$(GCC) $(OBJFLAGS) $< -o $@ -pg
#### --------------------------------------------------- ####

clean:
	rm -rf *.o *.a $(OUTDIR)
	rm -f candy
	rm -f gprof
	rm -f gmon.out profile.txt
