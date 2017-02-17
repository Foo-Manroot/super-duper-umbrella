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
} dim_t;



/* ------------------------- */
/* DEFINICIONES DE ETIQUETAS */
/* ------------------------- */

/**
 * Nivel máximo en el juego
 */
#define MAX_NV 3



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
