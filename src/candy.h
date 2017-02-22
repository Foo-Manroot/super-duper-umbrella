#ifndef _CANDY_H_
#define _CANDY_H_

#include "common.h"
#include "libutils.h"

#include <time.h>


/* ------------------------ */
/* DECLARACIÓN DE FUNCIONES */
/* ------------------------ */
int reservar_mem (Malla *malla);
int rellenar (Malla *malla);

void imprimir_malla (Malla malla);

#endif
