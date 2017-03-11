#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

#include "common.h"
#include "libutils.h"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

/**
 * Etiqueta para marcar elementos coincidentes en la matriz
 */
#define COINCIDE 1

/**
 * Etiqueta para marcar elementos que no coinciden en la matriz
 */
#define NO_COINCIDE 0

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


/**
 * Realiza la llamada al núcleo CUDA que usa memoria compartida 
 * y comprueba el código de error. Si hay algún error, devuelve (return) ERR_CUDA.
 */
#define KERNEL_COMP(err, nombre, bloques, hilos, tam, ...)			\
	imprimir (DETALLE_DEBUG,  "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n"		\
				"Lanzando el núcleo '%s' con las "		\
				"siguientes dimensiones: \n"			\
				"\tBloques: x=%d, y=%d\n"			\
				"\tHilos (por bloque): x=%d, y=%d, z=%d\n"	\
				"\tMemoria compartida: %d bytes\n"		\
				"+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n",		\
				#nombre,					\
				bloques.x, bloques.y,				\
				hilos.x, hilos.y, hilos.z,			\
				tam);						\
	nombre <<< (bloques), (hilos) , (tam) >>> (__VA_ARGS__);		\
										\
	if ((err = cudaGetLastError ()) != cudaSuccess)				\
	{									\
		imprimir (DETALLE_LOG, "Error en la línea %d de '%s': "		\
				"%s\n", __LINE__, __FILE__,			\
				cudaGetErrorString (err));			\
		return ERR_CUDA;						\
	}



/* --------------------------------------- */
/* DECLARACIÓN DE FUNCIONES DE DISPOSITIVO */
/* --------------------------------------- */

/**
 * Comprueba si es posible realizar un giro de 3x3 en la posición dada.
 *
 * @param posY
 * 		Coordenada Y del eje a comprobar.
 *
 * @param posX
 * 		Coordenada X del eje a comprobar.
 *
 * @param dimens
 * 		Dimensiones de la matriz a comprobar.
 * 
 *
 * @return
 *		true si es posible.
 *		false si no lo es.
 */
__device__ bool comprobar_giro (int posY, int posX, Dim dimens);

/**
 * Busca el primer elemento no vacío por encima de la posición especificada.
 * Además, este elemento se convierte a DIAMANTE_VACIO.
 *
 * @param matriz
 *		Matriz en la que se ha de buscar el elemento.
 *
 * @param fila_ini
 *		Fila del primer elemento a comprobar.
 *
 * @param columna
 *		Columna a comprobar.
 *
 * @param dimens
 *		Dimensiones de la matriz.
 *
 *
 * @return
 *		El primer elemento encontrado, si había alguno.
 *		-1 si no se encontró ningún elemento no vacío.
 */
__device__ int buscar_lleno (int *matriz, int fila_ini, int columna, Dim dimens);

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
__global__ void gen_aleat_cuda (curandState *estado,
				int *resultado,
				int min,
				int max);
/**
 * Mueve todos los elementos a la izquierda de fila_bomba hacia su derecha. Cuando llega
 * al primer elemento, genera un nuevo elemento.
 *
 * @param semilla
 *		Elemento inicial para generar la secuencia.
 *
 * @param resultado
 *		Vector en el que se almacenarán los números generados.
 *
 * @param min
 *		Límite inferior para generar un número (inclusivo).
 *
 * @param max
 *		Límite superior para generar un número (inclusivo).
 *
 * @param dimens
 *		Dimensiones de la matriz resultado.

 *
 * @param fila_bomb
 *		Fila a eliminar.
 */
__global__ void eliminar_fila_cuda (unsigned long semilla,
				    int *resultado,
				    const int min,
				    const int max,
				    const Dim dimens,
				    int fila_bomba);

/**
 * Mueve todos los elementos a la izquierda de fila_bomba hacia su derecha. Cuando llega
 * al primer elemento, genera un nuevo elemento.
 *
 * @param semilla
 *		Elemento inicial para generar la secuencia.
 *
 * @param resultado
 *		Vector en el que se almacenarán los números generados.
 *
 * @param min
 *		Límite inferior para generar un número (inclusivo).
 *
 * @param max
 *		Límite superior para generar un número (inclusivo).
 *
 * @param dimens
 *		Dimensiones de la matriz resultado.
 *
 *
 * @param fila_bomb
 *		Fila a eliminar.
 */
__global__ void eliminar_columna_cuda (unsigned long semilla,
				       int *resultado,
				       const int min,
				       const int max,
				       const Dim dimens,
				       int col_bomba);
/**
 * Gira todos los elementos posibles en grupos de 3x3 (bomba III).
 *
 * @param resultado
 *		Vector que almacena la matriz que va a ser cambiada.
 *
 * @param dimens
 * 		Dimensiones de la matriz.
 */
__global__ void girar_matriz_cuda (int *resultado, Dim dimens);


/**
 * Comprueba si la fila contiene elementos repetidos.
 *
 * @param matriz
 *		Matriz con los valores actuales de los diamantes.
 *
 * @param dimens
 *		Estructura con las dimensiones de la matriz.
 *
 * @param coincidencias
 *		Matriz en la que se va a indicar si había alguna coincidencia.
 */
__global__ void busar_coinc_cuda_fila (int *matriz,
				       Dim dimens,
				       int *coincidencias);

/**
 * Comprueba si la columna contiene elementos repetidos.
 *
 * @param matriz
 *		Matriz con los valores actuales de los diamantes.
 *
 * @param dimens
 *		Estructura con las dimensiones de la matriz.
 *
 * @param coincidencias
 *		Matriz en la que se va a indicar si había alguna coincidencia.
 */
__global__ void busar_coinc_cuda_col (int *matriz,
				       Dim dimens,
				       int *coincidencias);

/**
 * Elimina todos los elementos que se haya visto que han coincidido.
 *
 *
 * @param matriz
 *		Matriz con los valores actuales de los diamantes.
 *
 * @param dimens
 *		Estructura con las dimensiones de la matriz.
 *
 * @param coincidencias
 *		Matriz con las coincidencias encontradas.
 */
__global__ void eliminar_coinc_cuda (int *matriz,
				     Dim dimens,
				     int *coincidencias);


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

/**
 * Función para ejecutar la bomba I (eliminar fila).
 *
 * @param fila_bomba
 *		Fila que se debe eliminar (poner a DIAMANTE_VACIO).
 *
 * @param malla
 *		Estructura con la información del juego.
 *
 *
 * @return
 *		SUCCESS si todo ha salido correctamente.
 *		CUDA_ERR si hubo algún error relacionado con CUDA.
 */
int bomba_fila (int fila_bomba, Malla *malla);


/**
 * Función para ejecutar la bomba II (eliminar columna).
 *
 * @param col_bomba
 *		Columna que se debe eliminar (poner a DIAMANTE_VACIO).
 *
 * @param malla
 *		Estructura con la información del juego.
 *
 *
 * @return
 *		SUCCESS si todo ha salido correctamente.
 *		CUDA_ERR si hubo algún error relacionado con CUDA.
 */
int bomba_columna (int col_bomba, Malla *malla);

/**
 * Función para ejecutar la bomba III (girar en grupos de 3x3).
 *
 * @param malla
 *		Estructura con toda la información del juego (matriz, nivel
 *	y dimensiones).
 */
int bomba_giro (Malla *malla);

/**
 * Busca coincidencias en la matriz y marca las casillas para ser eliminadas (las deja
 * como DIAMANTE_VACIO.
 *
 * @return
 *		SUCCESS si todo ha salido correctamente.
 *		ERR_CUDA si hubo algún error al obtener las características del
 *	dispositivo.
 *		ERR_TAM si la matriz especificada sobrepasa las capacidades del
 *	dispositivo.
 */
int eliminar_coincidencias (Malla *malla);

/**
 * Rellena los diamantes vacíos en la matriz.
 *
 * @return
 *		SUCCESS si todo ha salido correctamente.
 *		ERR_CUDA si hubo algún error al obtener las características del
 *	dispositivo.
 *		ERR_TAM si la matriz especificada sobrepasa las capacidades del
 *	dispositivo.
 */
int llenar_vacios (Malla *malla);

#endif
