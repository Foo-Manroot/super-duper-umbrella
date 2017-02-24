#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdbool.h>

/***************************************************************************************
 * 	En este archivo se definen todas las estructuras, variables y códigos de salida
 * que se usan en toda la aplicación.
 ***************************************************************************************/


/* --------------------------- */
/* DEFINICIONES DE ESTRUCTURAS */
/* --------------------------- */

/**
 * Estructura para las dimensiones de la matriz
 */
typedef struct {

	int filas;
	int columnas;
} Dim;

/**
 * Estructura Diamante
 */
typedef struct {
    
    //id = {1,2,3,4,5,6,7,8} indican el tipo de diamante
    //id = {0} indica que no hay fiamante en ese hueco
    int id;
   
} Diamante;

/**
 * Estructura para la matriz de juego
 */
typedef struct {

	int nivel;		/* Nivel actual */
	Dim dimens;		/* Dimensiones de la matriz */
	Diamante *matriz;	/* Matriz de juego */
} Malla;



/* ------------------------- */
/* DEFINICIONES DE ETIQUETAS */
/* ------------------------- */

/**
 * Nivel máximo en el juego
 */
#define MAX_NV 3

/**
 * Valor para representar un hueco en el tablero (diamante vacío)
 */
#define DIAMANTE_VACIO 0

/**
 * Mayor valor admitido para un diamante
 */
#define DIAMANTE_MAX 8



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
 * Error al pasar los argumentos a la función.
 */
#define ERR_ARGS -1

/**
 * Error al abrir un archivo.
 */
#define ERR_ARCHIVO -2

/**
 * Error al gestionar la memoria (reservar, liberar...)
 */
#define ERR_MEM -3


#endif
