#ifndef _LIBUTILS_H_
#define _LIBUTILS_H_

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

/**
 * Mensaje de ayuda para mostrar el funcionamiento del programa.
 */
#define MSG_AYUDA "\n\
Práctica de Ampliación de Programación Avanzada.\n\
	Daniel Estangüi y Miguel García\n\
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


/* ------------------------ */
/* DECLARACIÓN DE FUNCIONES */
/* ------------------------ */
int procesar_args (int argc, char *argv []);
void imprimir_info ();


#endif
