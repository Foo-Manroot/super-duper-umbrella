#include "include/kernel.cuh"

/* ------------------------ */
/* FUNCIONES DE DISPOSITIVO */
/* ------------------------ */

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
__device__ bool comprobar_giro (int posY, int posX, Dim dimens)
{
	int filas = dimens.filas,
	    cols = dimens.columnas;

	/* Comprueba los límites del eje de giro */
	if ( ((posY + 1) >= filas)
		|| ((posX + 1) >= cols) )
	{
		return false;
	}

	if(((posY - 1) == 0)
		|| ((posY - 1) % 3) == 0)
	{
		/* Posición correcta para el eje Y */
		if(((posX - 1) == 0)
			|| ((posX - 1) % 3) == 0)
		{
			/* Posición correcta para el eje X */
			return true;
		}

	}

	return false;

}

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

	/* Comprueba los límites de la matriz */
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

/**
 * Mueve todos los elementos a la izquierda de fila_bomba hacia su derecha. Cuando llega
 * al primer elemento, genera un nuevo elemento.
 *
 * @param semilla
 *		Elemento inicial para generar la secuencia (para crear los
 *	nuevos elementos).
 *
 * @param resultado
 *		Vector que almacena la matriz que va a ser cambiada.
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
__global__ void eliminar_fila (unsigned long semilla,
			       int *resultado,
			       const int min,
			       const int max,
			       const Dim dimens,
			       int fila_bomba)
{
	int fila = blockIdx.y * blockDim.y + threadIdx.y,
	    columna = blockIdx.x * blockDim.x + threadIdx.x,
	    i,
	    rand_int;

	curandState estado;

	float rand_f;

	if ((columna >= dimens.columnas)
		|| (fila != fila_bomba))
	{
		return;
	}

	/* Intercambia los elementos desde la fila actual hasta el principio */
	for (i = fila_bomba; i > 0; i--)
	{
		resultado [(i * dimens.columnas) + columna]
			= resultado [( (i - 1) * dimens.columnas ) + columna];
	}

	/* Genera el último elemento */
	curand_init (semilla, columna, 0, &estado);
	/* El número se genera primero con coma flotante (ajustando los
	límites como se haya especificado) y luego se convierte a entero.
	Esto es más rápido que realizar la operación de módulo */
	rand_f = curand_uniform (&estado);

	rand_f *= (max - min + 0.999999);
	rand_f += min;

	/* Convierte el float a entero */
	rand_int = __float2int_rz (rand_f);

	/* Guarda el resultado */
	resultado [columna] = rand_int;

}

/**
 * Mueve todos los elementos a la izquierda de fila_bomba hacia su derecha. Cuando llega
 * al primer elemento, genera un nuevo elemento.
 *
 * @param semilla
 *		Elemento inicial para generar la secuencia (para crear los
 *	nuevos elementos).
 *
 * @param resultado
 *		Vector que almacena la matriz que va a ser cambiada.
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
__global__ void eliminar_columna (unsigned long semilla,
				  int *resultado,
				  const int min,
				  const int max,
				  const Dim dimens,
				  int col_bomba)
{
	int fila = blockIdx.y * blockDim.y + threadIdx.y,
	    columna = blockIdx.x * blockDim.x + threadIdx.x,
	    i,
	    rand_int;

	curandState estado;

	float rand_f;

	if ((fila >= dimens.filas)
		|| (columna != col_bomba))
	{
		return;
	}

	/* Intercambia los elementos desde la fila actual hasta el principio */
	for (i = col_bomba; i > 0; i--)
	{
		resultado [(fila * dimens.columnas) + i]
			= resultado [(fila * dimens.columnas ) + i - 1];
	}

	/* Genera el último elemento */
	curand_init (semilla, fila, 0, &estado);
	/* El número se genera primero con coma flotante (ajustando los
	límites como se haya especificado) y luego se convierte a entero.
	Esto es más rápido que realizar la operación de módulo */
	rand_f = curand_uniform (&estado);

	rand_f *= (max - min + 0.999999);
	rand_f += min;

	/* Convierte el float a entero */
	rand_int = __float2int_rz (rand_f);

	/* Guarda el resultado */
	resultado [fila * dimens.columnas] = rand_int;
}

/**
 * Gira todos los elementos posibles en grupos de 3x3 (bomba III).
 *
 * @param resultado
 *		Vector que almacena la matriz que va a ser cambiada.
 *
 * @param dimens
 *		Dimensiones de la matriz.
 */
__global__ void girar_matriz (int *resultado, Dim dimens)
{
	int fila = blockIdx.y * blockDim.y + threadIdx.y,
	    columna = blockIdx.x * blockDim.x + threadIdx.x,
	    posY = fila - 1,
	    posX = columna - 1,
	    aux;

	if ((fila >= dimens.filas)
		|| (columna >= dimens.columnas))
	{
		return;
	}

	if (comprobar_giro (fila, columna, dimens))
	{
		/* Se realizan los intercambios de manera manual */
		aux = resultado [(posY * dimens.columnas) + posX];
		/* ---- */
		resultado [(posY * dimens.columnas) + posX]
			= resultado [( (posY + 1) * dimens.columnas) + posX];

		resultado [( (posY + 1) * dimens.columnas) + posX]
			= resultado [( (posY + 2) * dimens.columnas) + posX];

		resultado [( (posY + 2) * dimens.columnas) + posX]
			= resultado [( (posY + 2) * dimens.columnas) + posX + 1];

		/* ---- */
		resultado [( (posY + 2) * dimens.columnas) + posX + 1]
			= resultado [( (posY + 2) * dimens.columnas) + posX + 2];

		resultado [( (posY + 2) * dimens.columnas) + posX + 2]
			= resultado [( (posY + 1) * dimens.columnas) + posX + 2];

		resultado [( (posY + 1) * dimens.columnas) + posX + 2]
			= resultado [(posY * dimens.columnas) + posX + 2];

		/* ---- */
		resultado [(posY * dimens.columnas) + posX + 2]
			= resultado [(posY * dimens.columnas) + posX + 1];

		resultado [(posY * dimens.columnas) + posX + 1] = aux;
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

	max = max_nv (*malla);

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
		time (NULL), matriz_d, 1, max, malla->dimens
	);

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
int bomba_fila (int fila_bomba, Malla *malla)
{
	cudaError_t err;
	dim3 bloques,
	     hilos;

	int tam = malla->dimens.filas * malla->dimens.columnas,
	    i,
	    j,
	    idx = 0,
	    max = max_nv (*malla);

	int *matriz_d,
	    *aux = (int *) malloc (tam * sizeof aux [0]);

	/* Dimensiones para luego crear un hilo por columna */
	Dim dim_matr_hilos;

	dim_matr_hilos.filas = 1;
	dim_matr_hilos.columnas = malla->dimens.columnas;

	/* Inicializa la matriz auxiliar */
	for (i = 0; i < malla->dimens.filas; i++)
	{
		for (j = 0; j < malla->dimens.columnas; j++)
		{
			idx = (i * malla->dimens.columnas) + j;
			aux [idx] = malla->matriz [idx].id;
		}
	}

	/* Reserva memoria en el dispositivo y copia la matriz */
	CUDA (err,
		cudaMalloc ((void **) &matriz_d,
				tam * sizeof matriz_d [0])
	);

	CUDA (err,
		cudaMemcpy (matriz_d, aux, tam * sizeof matriz_d [0],
				cudaMemcpyHostToDevice)
	);

	/* Llama al núcleo para eliminar la fila */
	obtener_dim (&bloques, &hilos, dim_matr_hilos);

	KERNEL (err, eliminar_fila,
		bloques, hilos,
		time (NULL), matriz_d, 1, max, malla->dimens, fila_bomba
	);

	/* Copia la información de vuelta y libera la memoria en el dispositivo */
	CUDA (err,
		cudaMemcpy (aux, matriz_d, tam * sizeof aux [0], cudaMemcpyDeviceToHost)
	);

	/* Copiar directamente un array de Diamante desde el dispositivo da problemas,
	así que se usa un array de enteros para crear los números aleatorios en
	paralelo y luego la CPU se encarga de crear los elementos de tipo Diamante */
	copiar_matriz (aux, malla);

	CUDA (err,
		cudaFree (matriz_d)
	);

	return SUCCESS;
}

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
int bomba_columna (int col_bomba, Malla *malla)
{
	cudaError_t err;
	dim3 bloques,
	     hilos;

	int tam = malla->dimens.filas * malla->dimens.columnas,
	    i,
	    j,
	    idx = 0,
	    max = max_nv (*malla);

	int *matriz_d,
	    *aux = (int *) malloc (tam * sizeof aux [0]);

	/* Dimensiones para luego crear un hilo por columna */
	Dim dim_matr_hilos;

	dim_matr_hilos.filas = malla->dimens.filas;
	dim_matr_hilos.columnas = 1;

	/* Inicializa la matriz auxiliar */
	for (i = 0; i < malla->dimens.filas; i++)
	{
		for (j = 0; j < malla->dimens.columnas; j++)
		{
			idx = (i * malla->dimens.columnas) + j;
			aux [idx] = malla->matriz [idx].id;
		}
	}

	/* Reserva memoria en el dispositivo y copia la matriz */
	CUDA (err,
		cudaMalloc ((void **) &matriz_d,
				tam * sizeof matriz_d [0])
	);

	CUDA (err,
		cudaMemcpy (matriz_d, aux, tam * sizeof matriz_d [0],
				cudaMemcpyHostToDevice)
	);

	/* Llama al núcleo para eliminar la fila */
	obtener_dim (&bloques, &hilos, dim_matr_hilos);

	KERNEL (err, eliminar_columna,
		bloques, hilos,
		time (NULL), matriz_d, 1, max, malla->dimens, col_bomba
	);

	/* Copia la información de vuelta y libera la memoria en el dispositivo */
	CUDA (err,
		cudaMemcpy (aux, matriz_d, tam * sizeof aux [0], cudaMemcpyDeviceToHost)
	);

	/* Copiar directamente un array de Diamante desde el dispositivo da problemas,
	así que se usa un array de enteros para crear los números aleatorios en
	paralelo y luego la CPU se encarga de crear los elementos de tipo Diamante */
	copiar_matriz (aux, malla);

	CUDA (err,
		cudaFree (matriz_d)
	);

	return SUCCESS;
}

/**
 * Función para ejecutar la bomba III (girar en grupos de 3x3).
 *
 * @param malla
 *		Estructura con toda la información del juego (matriz, nivel
 *	y dimensiones).
 */
int bomba_giro (Malla *malla)
{
	cudaError_t err;
	dim3 bloques,
	     hilos;

	int tam = malla->dimens.filas * malla->dimens.columnas,
	    i,
	    j,
	    idx = 0,
	    max = max_nv (*malla);

	int *matriz_d,
	    *aux = (int *) malloc (tam * sizeof aux [0]);

	/* Inicializa la matriz auxiliar */
	for (i = 0; i < malla->dimens.filas; i++)
	{
		for (j = 0; j < malla->dimens.columnas; j++)
		{
			idx = (i * malla->dimens.columnas) + j;
			aux [idx] = malla->matriz [idx].id;
		}
	}

	/* Reserva memoria en el dispositivo y copia la matriz */
	CUDA (err,
		cudaMalloc ((void **) &matriz_d,
				tam * sizeof matriz_d [0])
	);

	CUDA (err,
		cudaMemcpy (matriz_d, aux, tam * sizeof matriz_d [0],
				cudaMemcpyHostToDevice)
	);

	/* Llama al núcleo para eliminar la fila */
	obtener_dim (&bloques, &hilos, malla->dimens);

	KERNEL (err, girar_matriz,
		bloques, hilos,
		matriz_d, malla->dimens
	);

	/* Copia la información de vuelta y libera la memoria en el dispositivo */
	CUDA (err,
		cudaMemcpy (aux, matriz_d, tam * sizeof aux [0], cudaMemcpyDeviceToHost)
	);

	/* Copiar directamente un array de Diamante desde el dispositivo da problemas,
	así que se usa un array de enteros para crear los números aleatorios en
	paralelo y luego la CPU se encarga de crear los elementos de tipo Diamante */
	copiar_matriz (aux, malla);

	CUDA (err,
		cudaFree (matriz_d)
	);

	return SUCCESS;
}
