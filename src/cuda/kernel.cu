#include "include/kernel.cuh"

/* ------- */
/* NÚCLEOS */
/* ------- */

/**
 * Genera un número aleatorio en base a la secuencia especificada y al índice del hilo.
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
 */
__global__ void generar_aleat (unsigned long semilla,
			       int *resultado,
			       const int min,
			       const int max,
			       const Dim dimens)
{
	int rand_int,
	    fila = blockIdx.y * blockDim.y + threadIdx.y,
	    columna = blockIdx.x * blockDim.x + threadIdx.x,
	    aux = (fila * dimens.columnas) + columna;

	curandState estado;

	if ((fila <= dimens.filas)
	   && (columna <= dimens.columnas))
	{
		curand_init (semilla, aux, 0, &estado);
		/* El número se genera primero con coma flotante (ajustando los
		límites como se haya especificado) y luego se convierte a entero.
		Esto es más rápido que realizar la operación de módulo */
		float rand_f = curand_uniform (&estado);

		rand_f *= (max - min + 0.999999);
		rand_f += min;

		/* Convierte el float a entero */
		rand_int = __float2int_rz (rand_f);

		/* Guarda el resultado */
		resultado [aux] = rand_int;
	}
}

/* -------------------- */
/* FUNCIONES AUXILIARES */
/* -------------------- */

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
int obtener_dim (dim3 *bloques, dim3 *hilos, Dim tam_matriz)
{
	cudaDeviceProp propiedades;
	cudaError_t err;
	int id_dispos = -1;

	/* Busca el dispositivo con versión >= 2 (para poder usar más hilos
	por bloque) */
	propiedades.major = 2;

	CUDA (err,
	cudaChooseDevice (&id_dispos, &propiedades)
	);
	/* Actualiza la información del dispositivo (chooseDevice no lo hizo
	correctamente) */
	CUDA (err,
	cudaGetDeviceProperties (&propiedades, id_dispos)
	);

	imprimir (DETALLE_DEBUG, "\n -> Escogido dispositivo %d, con versión %d.%d\n\n",
				id_dispos,
				propiedades.major, propiedades.minor);

	cudaSetDevice (id_dispos);

	/* Número constante de hilos por bloque (para versiones anteriores
	a Fermi, 16 hilos) */
	hilos->x = (propiedades.major < 2)? 16 : 32;
	hilos->y = (propiedades.major < 2)? 16 : 32;
	hilos->z = 1;

	/* Se calcula el número de bloques que se deben utilizar */
	bloques->x = ceil (((float) tam_matriz.columnas) / ((float) hilos->x));
	bloques->y = ceil (((float) tam_matriz.filas) / ((float) hilos->y));
	bloques->z = 1;

	/* Si la matriz no cabe, se avisa */
	if ((bloques->x > propiedades.maxGridSize [0])
		|| (bloques->y > propiedades.maxGridSize [1]))
	{
		imprimir (DETALLE_LOG, "\n -> Error: la matriz es demasiado grande "
					"para el dispositivo\n");
		return ERR_TAM;
	}

	return SUCCESS;
}


/**
 * Copia la información de la matriz de enteros (el resultado devuelto por el
 * dispositivo) en la matriz de juego con la que trabaja la CPU.
 *
 * @param matriz_d
 *		Matriz de enteros con los resultados de la tarjeta.
 *
 * @param malla
 *		Malla con la información del juego, cuya matriz va a ser actualizada.
 */
void copiar_matriz (int *matriz_d, Malla *malla)
{
	int i,
	    j,
	    idx,
	    filas = malla->dimens.filas,
	    columnas = malla->dimens.columnas;

	for (i = 0; i < filas; i++)
	{
		for (j = 0; j < columnas; j++)
		{
			idx = (i * columnas) + j;
			malla->matriz [idx] = crear_diamante (matriz_d [idx]);
		}
	}
}


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
int matriz_aleat (Malla *malla)
{
	int max = DIAMANTE_VACIO,
	    filas = malla->dimens.filas,
	    columnas = malla->dimens.columnas,
	    tam = filas * columnas;

	cudaError_t err;

	dim3 bloques,
	     hilos;

	int *matriz_d,
	    *aux = (int *) malloc (tam * sizeof aux [0]);

	switch (malla->nivel)
	{
		case 1:
			max = 4;
			break;

		case 2:
			max = 6;
			break;

		case 3:
			max = DIAMANTE_MAX;
			break;

		default:
			max = DIAMANTE_MAX;
	}

	/* Comprueba que la matriz tiene memoria reservada */
	if (malla->matriz == NULL)
	{
		imprimir (DETALLE_DEBUG,
			  "Error: la matriz no tiene memoria reservada.\n");
		return ERR_MEM;
	}

	CUDA (err,
		cudaMalloc ((void **) &matriz_d, tam * sizeof matriz_d [0])
	);

	/* Llama al núcleo para inicializar la secuencia de números aleatorios */
	obtener_dim (&bloques, &hilos, malla->dimens);

	/* Genera los números aleatorios y los copia en la matriz */
	KERNEL (err, generar_aleat,
		bloques, hilos,
		time (NULL), matriz_d, 1, max, malla->dimens);

	/* Usa la matriz auxiliar con los números aleatorios para crear los diamantes */
	CUDA (err,
		cudaMemcpy (aux, matriz_d, tam * sizeof aux [0], cudaMemcpyDeviceToHost)
	);

	/* Copiar directamente un array de Diamante desde el dispositivo da problemas,
	así que se usa un array de enteros para crear los números aleatorios en
	paralelo y luego la CPU se encarga de crear los elementos de tipo Diamante */
	copiar_matriz (aux, malla);

	/* Se libera la memoria del dispositivo */
	CUDA (err,
		cudaFree (matriz_d)
	);

	return SUCCESS;
}


/**
 * Función para ejecutar la bomba I (eliminar fila).
 *
 * @param malla
 *		Estructura con la información del juego.
 *
 * @param fila
 *		Fila que se debe eliminar (poner a DIAMANTE_VACIO).
 *
 *
 * @return
 *		SUCCESS si todo ha salido correctamente.
 *		CUDA_ERR si hubo algún error relacionado con CUDA.
 */
//int bomba_fila (Malla malla, int fila)
//{
//	
//
//	return SUCCESS;
//}

