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
__device__ int buscar_lleno (int *matriz, int fila_ini, int columna, Dim dimens)
{
	int elem = -1,
	    fila = fila_ini,
	    aux;

	while ( (elem == -1)
		&& (fila > 0))
	{
		aux = (fila * dimens.columnas) + columna;

		if (matriz [aux] != DIAMANTE_VACIO)
		{
			elem = matriz [aux];
			matriz [aux] = DIAMANTE_VACIO;
		}

		fila--;
	}

	return elem;
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
__global__ void gen_aleat_cuda (unsigned long semilla,
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
__global__ void eliminar_fila_cuda (unsigned long semilla,
				    int *resultado,
				    const int min,
				    const int max,
				    const Dim dimens,
				    int fila_bomba)
{
	int columna = blockIdx.x * blockDim.x + threadIdx.x,
	    i,
	    rand_int;

	curandState estado;

	float rand_f;

	if (columna >= dimens.columnas)
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
__global__ void eliminar_columna_cuda (unsigned long semilla,
				       int *resultado,
				       const int min,
				       const int max,
				       const Dim dimens,
				       int col_bomba)
{
	int fila = blockIdx.y * blockDim.y + threadIdx.y,
	    i,
	    rand_int;

	curandState estado;

	float rand_f;

	if (fila >= dimens.filas)
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
__global__ void girar_matriz_cuda (int *resultado,
				   Dim dimens)
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
			= resultado [( (posY + 2) * dimens.columnas) + posX];

		resultado [( (posY + 2) * dimens.columnas) + posX]
			= resultado [( (posY + 2) * dimens.columnas) + posX + 2];

		resultado [( (posY + 2) * dimens.columnas) + posX + 2]
			= resultado [(posY * dimens.columnas) + posX + 2];

		resultado [(posY * dimens.columnas) + posX + 2] = aux;

		/* ---- */
		aux = resultado [(posY * dimens.columnas) + posX + 1];

		resultado [(posY * dimens.columnas) + posX + 1]
			= resultado [( (posY + 1) * dimens.columnas) + posX];

		resultado [( (posY + 1) * dimens.columnas) + posX]
			= resultado [( (posY + 2) * dimens.columnas) + posX + 1];

		resultado [( (posY + 2) * dimens.columnas) + posX + 1]
			= resultado [( (posY + 1) * dimens.columnas) + posX + 2];

		resultado [( (posY + 1) * dimens.columnas) + posX + 2] = aux;
	}
}

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
				       int *coincidencias)
{
	int fila = blockIdx.y * blockDim.y + threadIdx.y,
	    i,
	    aux = fila * dimens.columnas;

	if (fila >= dimens.filas)
	{
		return;
	}

	/* Recorre la matriz marcando los elementos iguales consecutivos */
	for (i = 0; i < (dimens.columnas - 2) ; i++)
	{
		if ( (matriz [aux + i] == matriz [aux + i + 1])
			&& (matriz [aux + i] == matriz [aux + i + 2]) )
		{
			coincidencias [aux + i] = COINCIDE;
			coincidencias [aux + i + 1] = COINCIDE;
			coincidencias [aux + i + 2] = COINCIDE;
		}
	}
}

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
				      int *coincidencias)
{
	int columna = blockIdx.x * blockDim.x + threadIdx.x,
	    i;

	if (columna >= dimens.columnas)
	{
		return;
	}

	/* Recorre la matriz marcando los elementos iguales consecutivos */
	for (i = 0; i < (dimens.filas - 2) ; i++)
	{
		if ( (matriz [(i * dimens.columnas) + columna]
			== matriz [( (i + 1) * dimens.columnas) + columna])
			&& (matriz [(i * dimens.columnas) + columna]
				== matriz [( (i + 2) * dimens.columnas) + columna]) )
		{
			coincidencias [(i * dimens.columnas) + columna] = COINCIDE;
			coincidencias [(i + 1) * dimens.columnas + columna] = COINCIDE;
			coincidencias [(i + 2) * dimens.columnas + columna] = COINCIDE;
		}
	}
}


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
				     int *coincidencias)
{
	int fila = blockIdx.y * blockDim.y + threadIdx.y,
	    columna = blockIdx.x * blockDim.x + threadIdx.x;

	if ( (fila >= dimens.filas)
		|| (columna >= dimens.columnas))
	{
		return;
	}

	if (coincidencias [(fila * dimens.columnas) + columna] == COINCIDE)
	{
		matriz [(fila * dimens.columnas) + columna] = DIAMANTE_VACIO;
	}
}

/**
 * Comprueba todos los huecos de la columna y rellena los vacíos.
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
__global__ void llenar_vacios_cuda (unsigned long semilla,
				    int *resultado,
				    const int min,
				    const int max,
				    const Dim dimens)
{
	int columna = blockIdx.x * blockDim.x + threadIdx.x,
	    i,
	    elem,
	    rand_int;

	curandState estado;
	float rand_f;

	if (columna >= dimens.columnas)
	{
		return;
	}

	/* Recorre la columna hasta encontrar un elemento vacío */
	for (i = dimens.filas; i >= 0; i--)
	{
		elem = resultado [(i * dimens.columnas) + columna];

		if (elem == DIAMANTE_VACIO)
		{
			/* Busca el primer elemento que haya por encima y lo baja */
			elem = buscar_lleno (resultado, i, columna, dimens);

			if (elem == -1)
			{
				curand_init (semilla, i + columna, 0, &estado);
				/* El número se genera primero con coma flotante
				(ajustando los límites como se haya especificado) y
				luego se convierte a entero. Esto es más rápido que
				realizar la operación de módulo */
				rand_f = curand_uniform (&estado);

				rand_f *= (max - min + 0.999999);
				rand_f += min;

				/* Convierte el float a entero */
				rand_int = __float2int_rz (rand_f);

				/* Guarda el resultado */
				elem = rand_int;
			}

			resultado [(i * dimens.columnas) + columna] = elem;
		}
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

	imprimir (DETALLE_EXTRA, "\n -> Escogido dispositivo %d, con versión %d.%d\n\n",
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

	imprimir (DETALLE_EXTRA, "Se usan bloques de %d x %d para alojar los (%d x %d)"
				 " hilos necesarios.\n",
				hilos->x, hilos->y,
				tam_matriz.filas, tam_matriz.columnas);

	/* Si la matriz no cabe, se avisa */
	if ((bloques->x > propiedades.maxGridSize [0])
		|| (bloques->y > propiedades.maxGridSize [1]))
	{
		imprimir (DETALLE_LOG, "\n -> Error: la matriz es demasiado grande "
					"para el dispositivo\n");
		return ERR_TAM;
	}

	/* Limitación para la práctica. Si la matriz cabe en un bloque, se divide para
	que ocupe 4 */
	if ((tam_matriz.columnas * tam_matriz.filas) < propiedades.maxThreadsPerBlock)
	{
		hilos->x = ceil ( ((float) tam_matriz.columnas) / 2.0 );
		hilos->y = ceil ( ((float) tam_matriz.filas) / 2.0 );

		bloques->x = ceil (((float) tam_matriz.columnas) / ((float) hilos->x));
		bloques->y = ceil (((float) tam_matriz.filas) / ((float) hilos->y));

		imprimir (DETALLE_EXTRA, " --> Limitación artificial (para la"
					" práctica): se usan %d x %d bloques de "
					" %d x %d hilos. La matriz es de %d x %d "
					" elementos.\n",
					 bloques->x, bloques->y,
					 hilos->x, hilos->y,
					 tam_matriz.filas, tam_matriz.columnas);
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
	KERNEL (err, gen_aleat_cuda,
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
 *		ERR_CUDA si hubo algún error al obtener las características del
 *	dispositivo.
 *		ERR_TAM si la matriz especificada sobrepasa las capacidades del
 *	dispositivo.
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

	KERNEL (err, eliminar_fila_cuda,
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
 *		ERR_CUDA si hubo algún error al obtener las características del
 *	dispositivo.
 *		ERR_TAM si la matriz especificada sobrepasa las capacidades del
 *	dispositivo.
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

	/* Llama al núcleo para eliminar la columna */
	obtener_dim (&bloques, &hilos, dim_matr_hilos);

	KERNEL (err, eliminar_columna_cuda,
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
 *
 *
 * @return
 *		SUCCESS si todo ha salido correctamente.
 *		ERR_CUDA si hubo algún error al obtener las características del
 *	dispositivo.
 *		ERR_TAM si la matriz especificada sobrepasa las capacidades del
 *	dispositivo.
 */
int bomba_giro (Malla *malla)
{
	cudaError_t err;
	dim3 bloques,
	     hilos;

	int tam = malla->dimens.filas * malla->dimens.columnas,
	    i,
	    j,
	    idx = 0;

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

	/* Llama al núcleo para girar la matriz */
	obtener_dim (&bloques, &hilos, malla->dimens);

	KERNEL (err, girar_matriz_cuda,
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
int eliminar_coincidencias (Malla *malla)
{
	cudaError_t err;
	dim3 bloques,
	     hilos;

	int tam = malla->dimens.filas * malla->dimens.columnas,
	    i,
	    j,
	    idx = 0;

	int *matriz_d,
	    *coincidencias_d,
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
		cudaMalloc ((void **) &coincidencias_d,
				tam * sizeof coincidencias_d [0])
	);

	CUDA (err,
		cudaMemset (coincidencias_d, NO_COINCIDE,
				tam * sizeof coincidencias_d [0])
	);

	CUDA (err,
		cudaMemcpy (matriz_d, aux, tam * sizeof matriz_d [0],
				cudaMemcpyHostToDevice)
	);

	/* Llama al núcleo para comprobar la matriz */
	obtener_dim (&bloques, &hilos, dim_matr_hilos);

	KERNEL (err, busar_coinc_cuda_fila,
		bloques, hilos,
		matriz_d, malla->dimens, coincidencias_d
	);

	dim_matr_hilos.filas = 1;
	dim_matr_hilos.columnas = malla->dimens.columnas;
	obtener_dim (&bloques, &hilos, dim_matr_hilos);

	KERNEL (err, busar_coinc_cuda_col,
		bloques, hilos,
		matriz_d, malla->dimens, coincidencias_d
	);

	/* Utiliza la matriz con los elementos marcados para eliminarlos */
	obtener_dim (&bloques, &hilos, malla->dimens);
	KERNEL (err, eliminar_coinc_cuda,
		bloques, hilos,
		matriz_d, malla->dimens, coincidencias_d
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

	CUDA (err,
		cudaFree (coincidencias_d)
	);

	return SUCCESS;
}


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
int llenar_vacios (Malla *malla)
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

	/* Llama al núcleo para comprobar la matriz */
	obtener_dim (&bloques, &hilos, dim_matr_hilos);

	KERNEL (err, llenar_vacios_cuda,
		bloques, hilos,
		time (NULL), matriz_d, 1, max, malla->dimens
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
