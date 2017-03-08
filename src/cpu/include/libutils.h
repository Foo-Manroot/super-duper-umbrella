#ifndef _LIBUTILS_H_
#define _LIBUTILS_H_

#include "common.h"
#include "menu.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>


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

/**
 * Colores
 */
#define NOR "\x1B[0m" //Color normal
#define ROJ "\x1B[31m"
#define VER "\x1B[32m"
#define AMA "\x1B[33m"
#define AZU "\x1B[34m"
#define ROS "\x1B[35m"
#define CYN "\x1B[36m"
#define BLA "\x1B[30m"
#define RST "\x1B[0m" //RESET

/**
 * Niveles de detalle
 */
#define DETALLE_LOG	0
#define DETALLE_DEBUG	1
#define DETALLE_EXTRA	2

/* ------------------------ */
/* DECLARACIÓN DE FUNCIONES */
/* ------------------------ */
int procesar_args (int argc, char *argv []);
void imprimir_info ();

Malla ver_params ();
int guardar (Malla malla, const char *nombre_fichero);

int cargar (Malla *malla, const char *nombre_fichero);
int reservar_mem (Malla *malla);

int rellenar (Malla *malla);
void mostrar_malla (Malla malla);

Diamante crear_diamante (int num);
void imprimir (int detalle, const char *cadena, ...);

void cambiar_params (Malla malla);

#endif
