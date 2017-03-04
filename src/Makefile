# Makefile para ejecutar las versiones de CPU y CUDA

# Llama a los Makefile de los directorios
all: 
	$(MAKE) -C cpu all
	$(MAKE) -C cuda all

####
# Objetivos para las implementaciones: compila todo, ejecuta el programa, y elimina
# los archivos producidos en la compilación cuando se sale

# CPU
cpu: .CPU
.CPU:
	$(MAKE) -C cpu all
	./cpu/candy -f 10 -c 10 -n 2 -vv
	$(MAKE) -C cpu clean

# CUDA
cuda: .CUDA
.CUDA:
	$(MAKE) -C cuda all
	./cuda/candy -f 10 -c 10 -n 2 -vv
	$(MAKE) -C cuda clean

# Llama a la regla 'clean' de los Makefile de los directorios
clean:
	$(MAKE) -C cpu clean
	$(MAKE) -C cuda clean
