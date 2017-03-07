#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

#include "common.h"
#include "libutils.h"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

/**
 * Realiza la llamada a la función CUDA y comprueba el valor devuelto. Si hay algún
 * error, devuelve (return) ERR_CUDA.
 */
#define CUDA(err, x) do { if ((err = (x)) != cudaSuccess) {	\
	imprimir (DETALLE_LOG, "Error en la línea %d de '%s': %s\n",	\
			__LINE__, __FILE__, cudaGetErrorString (err)); \
	return ERR_CUDA;}} while(0)


/**
 * Realiza la llamada al núcleo CUDA y comprueba el código de error. Si hay algún
 * error, devuelve (return) ERR_CUDA.
 */
#define KERNEL(err, nombre, bloques, hilos, ...)				\
	imprimir (DETALLE_DEBUG,  "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n"		\
				"Lanzando el núcleo '%s' con las "		\
				"siguientes dimensiones: \n"			\
				"\tBloques: x=%d, y=%d\n"			\
				"\tHilos (por bloque): x=%d, y=%d, z=%d\n"	\
				"+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n",		\
				#nombre,					\
				bloques.x, bloques.y,				\
				hilos.x, hilos.y, hilos.z);			\
	nombre <<< (bloques), (hilos) >>> (__VA_ARGS__);			\
										\
	if ((err = cudaGetLastError ()) != cudaSuccess)				\
	{									\
		imprimir (DETALLE_LOG, "Error en la línea %d de '%s': "		\
				"%s\n", __LINE__, __FILE__,			\
				cudaGetErrorString (err));			\
		return ERR_CUDA;						\
	}


/* ---------------------- */
/* DECLARACIÓN DE NÚCLEOS */
/* ---------------------- */

/**
 * Genera un número aleatorio en base a la secuencia especificada y al índice del hilo.
 *
 * @param estado
 *		Estado utilizado para generar la secuencia.
 *
 * @param resultado
 *		Vector en el que se almacenará el número generado.
 *
 * @param min
 *		Límite inferior para generar un número (inclusivo).
 *
 * @param max
 *		Límite superior para generar un número (inclusivo).
 */
__global__ void generar_aleat (curandState *estado,
			       int *resultado,
			       int min,
			       int max);

/* ----------------------------------- */
/* DECLARACIÓN DE FUNCIONES AUXILIARES */
/* ----------------------------------- */


/**
 * Rellena la matriz de juego con diamantes aleatorios.
 *
 * @param malla
 * 		Estructura de tipo Malla (definida en 'common.h') con las dimensiones de
 * 	la matriz y su contenido.
 *
 *
 * @return
 * 		SUCCESS si todo ha salido correctamente.
 * 		ERR_CUDA si alguna función CUDA ha fallado.
 */
int matriz_aleat (Malla *malla);

/**
 * Obtiene las dimensiones de los hilos necesarias para ejecutar la matriz con las
 * dimensiones especificadas, teniendo en cuenta las limitaciones del dispositivo.
 *
 * @param bloques
 *		Elemento de tipo dim3 para almacenar las dimensiones de los bloques
 *	dentro de la rejilla (2D).
 *
 * @param hilos
 *		Elemento de tipo dim3 para almacenar las dimensiones de los hilos dentro
 *	de los bloques (3D).
 *
 * @param tam_matriz
 *		Estructura Dim (definida en 'commno.h') con las dimensiones de la matriz
 *	que se desea usar en el dispositivo.
 *
 *
 * @return
 *		SUCCESS si todo ha salido correctamente.
 *		ERR_CUDA si hubo algún error al obtener las características del
 *	dispositivo.
 *		ERR_TAM si la matriz especificada sobrepasa las capacidades del
 *	dispositivo.
 */
int obtener_dim (dim3 bloques, dim3 hilos, Dim tam_matriz);


#endif
