# Makefile para ejecutar las versiones de CPU y CUDA

# Llama a los Makefile de los directorios (sin ejecutar nada)
all: 
	@echo "\n-------------------------"
	@echo   "**** COMPILANDO TODO ****"
	@echo   "-------------------------\n"
	@echo "\n **** PRIMER OBJETIVO: CPU **** \n"
	$(MAKE) -C cpu all
	@echo "\n **** SEGUNDO OBJETIVO: GPU (CUDA) **** \n"
	$(MAKE) -C cuda all
	@echo "\n **** TERCER OBJETIVO: CUDA con optimizaciones **** \n"
	$(MAKE) -C optimizada all
	@echo "\n **** CUARTO OBJETIVO: CUDA con optimizaciones y OpenGL **** \n"
	$(MAKE) -C openGL all

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

# Versión optimizada
optimizada: .optimizada
.optimizada:
	$(MAKE) -C optimizada all
	./optimizada/candy -f 10 -c 10 -n 2 -vv
	$(MAKE) -C optimizada clean


# Versión con OpenGL
openGL: .openGL
.openGL:
	$(MAKE) -C openGL all
	./openGL/candy -f 10 -c 10 -n 2 -vv
	$(MAKE) -C openGL clean


# Llama a la regla 'clean' de los Makefile de los directorios
clean:
	@echo "\n--------------------------------------"
	@echo   "++++ LIMPIANDO ARCHIVOS GENERADOS ++++"
	@echo   "--------------------------------------\n"
	@echo "\n ++++ Carpeta cpu/ ++++ \n"
	$(MAKE) -C cpu clean
	@echo "\n ++++ Carpeta cuda/ ++++ \n"
	$(MAKE) -C cuda clean
	@echo "\n ++++ Carpeta optimizada/ ++++ \n"
	$(MAKE) -C optimizada clean
	@echo "\n ++++ Carpeta openGL/ ++++ \n"
	$(MAKE) -C openGL clean
