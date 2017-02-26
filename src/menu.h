#ifndef _MENU_H_
#define _MENU_H_

#include "candy.h"
#include "common.h"

#include <string.h>

/* ------------------------ */
/* DECLARACIÓN DE ETIQUETAS */
/* ------------------------ */

/**
 * Mensaje para mostrar las opciones del menú principal
 */
#define MSG_MENU "\n\
---------------------\n\
Opciones disponibles:\n\
	1.- Mover diamante\n\
	2.- Bomba\n\
	3.- Guardar partida\n\
	4.- Cargar partida\n\
	5.- Cambiar nivel\n\
---------------------\n\
Introduzca la opción seleccionada: "

/* ------------------------ */
/* DECLARACIÓN DE FUNCIONES */
/* ------------------------ */
void menu (Malla malla);
int pedir_opcion (int min, int max);


void guardar_partida (Malla malla);
void cargar_partida (Malla *malla);

#endif
