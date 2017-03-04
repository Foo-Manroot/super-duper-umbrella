GCC=gcc
NVCC=nvcc
CFLAGS=-Wall
OBJFLAGS=-Wall -c
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
#cuda: joyas.cu
#	$(NVCC) joyas.cu -o joyas-CUDA
#----

####
candy: $(OUTDIR)/candy.o $(OUTDIR)/libutils.a
	$(GCC) $(CFLAGS) $^ -o $@

$(OUTDIR)/candy.o: candy.c $(INCDIR)/candy.h $(INCDIR)/common.h
	$(crear_dir)
	$(GCC) $(OBJFLAGS) $< -o $@
#----

####
# Biblioteca para el intérprete de comandos
$(OUTDIR)/libutils.a: $(OUTDIR)/utils.o $(OUTDIR)/menu.o
	$(crear_dir)
	$(LIB) $@ $^

$(OUTDIR)/utils.o: utils.c $(INCDIR)/libutils.h $(INCDIR)/common.h
	$(crear_dir)
	$(GCC) $(OBJFLAGS) $< -o $@

$(OUTDIR)/menu.o: menu.c $(INCDIR)/menu.h $(INCDIR)/common.h
	$(crear_dir)
	$(GCC) $(OBJFLAGS) $< -o $@
#----

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
####
