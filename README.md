# super-duper-umbrella

Este proyecto es el resultado de la práctica 1 de Ampliación de Programación Avanzada, en el que se nos pidió realizar un juego en el que se pretenden alinear elementos de la matriz (diamantes) para eliminarlos y que aparezcan otros nuevos.

Para la práctica se pidieron dos implementaciones: una en CPU (secuencial) y otra con CUDA (paralelizando todas las tareas posibles).

En base a la parte paralela, se utilizan algunas técnicas de optimización (uso de memoria compartida y desenrrollado de bucles) y, tras ello, se añade una interfaz gráfica con OpenGL (usando la biblioteca FreeGLUT).

Se puede leer el principio de 'Memoria.qwer' para obtener más detalles sobre la implementación.

## Dependencias
La implementación con CPU no tiene más dependencias que las bibliotecas estándar; así que sólo hace falta un compilador para C. Por defecto, el Makefile utiliza _gcc_; pero se puede cambiar de compilador cambiando el valor de la variable _GCC_. Por ejemplo, si se quisiera usar el compilador _cc_, se ejecutaría `make GCC=cc`

Para compilar los archivos que usen CUDA hacen falta las bibliotecas y herramientas proporcionadas con el [cuda-toolkit](https://developer.nvidia.com/cuda-toolkit). Si no están instaladas, el sistema no encontrará el compilador usado, _nvcc_
Para ejecutar cualquier cosa que no sea cpu/candy, hace falta una tarjeta que soporte CUDA (las de NVidia).

## Compilación
Para compilar hay un Makefile en cada carpeta con las distintas implementaciones. Para compilar cada una de estas subcarpetas, simplemente hay que ejecutar `make`.

Además, en la carpeta raíz (_src/_) hay un Makefile para compilar y ejecutar cualquiera de las implementaciones.
  - Si se quiere compilar todo, se debe ejecutar `make`.
  - Si se quieren limpiar todos los archivos generados al compilar (incluyendo el ejecutable), se debe ejecutar `make clean`
  - Si se quiere compilar, ejecutar y limpiar los archivos (cuando se salga del juego) de la implementación en CPU, se debe ejecutar `make cpu`. Lo mismo se aplica a la implementación con CUDA (`make cuda`), a la optimizada (maka `optimizada`) y a la versión con OpenGL (`make openGL`).
  
## Ejecución

Para ejecutar cualquiera de las versiones, simplemente se debe usar `./candy`. Si se quieren ver las opciones reconocidas, se puede usar `./candy -h`. La salida es la siguiente:

		Práctica de Ampliación de Programación Avanzada.
		Daniel Estangüi y Miguel García
	Uso correcto:
	./candy [-hman:f:c:v]
		-h
			Muestra este mensaje de ayuda y sale del programa
		-a | -m
			Habilita la ejecución automática (-a) o manual (-m). Si no se especifica, se habilita la ejecución automática por defecto. Estas opciones son excluyentes
		-n <nivel>
			Si se especifica, establece el nivel de inicio (del 1 al 3)
		-f <nº_filas>
			Establece el número de filas de la matriz de juego
		-c <nº_columnas>
			Establece el número de columnas de la matriz de juego
		-v
			Incrementa el nivel de detalle

Para jugar en las versiones en consola se presenta un menú para seleccionar la opción adecuada.

En la versión con OpenGL se pueden seleccionar dos casillas para intercambiarlas. Además, se pueden usar las siguientes teclas:
  - g: Guarda la partida en el archivo por defecto
  - c: Carga la partida del archivo por defecto
  - 91_x_: Elimina la fila _x_ (bomba I)
  - 92_x_: Elimina la columna _x_ (bomba II)
  - 93: Gira las casillas en grupos de 3x3 (bomba III)
