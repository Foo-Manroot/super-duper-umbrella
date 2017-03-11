#ifndef _CANDY_H_
#define _CANDY_H_

#include "common.h"
#include "libutils.h"
#include "kernel.cuh"

#include <sys/time.h>

/**
 * Movimientos de diamantes posibles
 */
#define MOV_DER 0
#define MOV_ABAJO 1
#define MOV_IZQ 2
#define MOV_ARRIBA 3

/* ------------------------ */
/* DECLARACIÓN DE FUNCIONES */
/* ------------------------ */
void mover_diamante(int posY, int posX, int mov, Malla malla);

int es_valido(int posY, int posX, int mov, Malla malla);

#endif
