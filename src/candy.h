#ifndef _CANDY_H_
#define _CANDY_H_

#include "common.h"
#include "libutils.h"

/* ------------------------ */
/* DECLARACIÓN DE FUNCIONES */
/* ------------------------ */
void recorrer_malla_giro (Malla malla);
void girar_matriz (int ejeY, int ejeX, Malla malla);

int es_posible_giro (int posY, int posX, Malla malla);
void eliminar_columna (int columna,Malla malla);

void recorrer_malla_reorden (Malla malla);
void reordenar_tablero (int posY,int posX,Malla malla);

void eliminar_fila (int fila, Malla malla);
void recorrer_malla_huecos (Malla malla);

void tratar_huecos(int posY, int posX, Malla malla);
void recorrer_malla_coincidencias (Malla malla);

void tratar_coincidencias(int posY, int posX, Malla malla);
void eliminar_coincidencias_eje(int posY, int posX,int eje,Malla malla);

void eliminar_coincidencias(int posY,int posX,int sen,Malla malla);
int buscar_coincidencias(int posY, int posX, int sen,Malla malla);

int son_iguales(Diamante d1,Diamante d2);
void mover_diamante(int posY, int posX, int mov, Malla malla);

int es_valido(int posY, int posX, int mov, Malla malla);

Diamante generar_diamante();

#endif
