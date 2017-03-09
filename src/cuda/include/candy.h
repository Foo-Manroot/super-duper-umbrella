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
void recorrer_malla (Malla malla);

void recorrer_malla_reorden (Malla malla);
void reordenar_tablero (int posY,int posX,Malla malla);

void recorrer_malla_huecos (Malla malla);

void tratar_huecos(int posY, int posX, Malla malla);

void mover_diamante(int posY, int posX, int mov, Malla malla);

int es_valido(int posY, int posX, int mov, Malla malla);
Diamante generar_diamante();

#endif
