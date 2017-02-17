#ifndef _JOYAS_H_
#define _JOYAS_H_

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

/**
 * Mensaje de ayuda para mostrar el funcionamiento del programa.
 */
#define MSG_AYUDA "\n\
Práctica de Ampliación de Programación Avanzada.\n\
	Daniel Estangui y Miguel García\n\
Uso correcto:\n\
joyas [-hman:f:c:v]\n\
	-h\n\
		Muestra este mensaje de ayuda y sale del programa\n\
	-a | -m\n\
		Habilita la ejecución automática (-a) o manual (-m). Si no se\
 especifica, se habilita la ejecución automática por defecto. Estas opciones son\
 excluyentes\n\
	-n <nivel>\n\
		Si se especifica, establece el nivel de inicio (del 1 al 3)\n\
	-f <nº_filas>\n\
		Establece el número de filas de la matriz de juego\n\
	-c <nº_columnas>\n\
		Establece el número de columnas de la matriz de juego\n\
	-v\n\
		Incrementa el nivel de detalle\n\
"

/**
 * Nivel máximo en el juego
 */
#define MAX_NV 3

/* --------------------------- */
/* DEFINICIONES DE ESTRUCTURAS */
/* --------------------------- */

/**
 * Estructura para las dimensiones de la matriz
 */
typedef struct {

	int filas;
	int columnas;
} dim_t;


/* ----------------------- */
/* DEFINICIONES DE ERRORES */
/* ----------------------- */

/**
 * La función acabó sin problemas.
 */
#define SUCCESS 0

/**
 * Éxito al procesar un argumento que requiere que se termine la ejecución (por ejemplo,
 * la opción '-h'.
 */
#define SUCC_ARGS 1

/**
 * Error al pasar los argumentos.
 */
#define ERR_ARGS -1


#endif
