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
		&& (fila >= 0))
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
	    rand_int,
	    aux;
	curandState estado;

	float rand_f;
	extern __shared__ int matriz_comp [];

	if ( (columna >= dimens.columnas)
		|| ( (blockIdx.y * blockDim.y + threadIdx.y) != 0) )
	{
		return;
	}

	/* Copia la columna en la memoria compartida */
	for (i = 0; i <= fila_bomba; i++)
	{
		aux = (i * dimens.columnas) + columna;
		matriz_comp [aux] = resultado [aux];
	}

	/* ---- A partir de aquí, trabaja con la memoria compartida ---- */

	/* Intercambia los elementos desde la fila actual hasta el principio */
	for (i = fila_bomba; i > 0; i--)
	{
		matriz_comp [(i * dimens.columnas) + columna]
			= matriz_comp [(i - 1) * dimens.columnas + columna];
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
	matriz_comp [columna] = rand_int;

	/* Copia los datos de vuelta a la memoria global */
	for (i = 0; i <= fila_bomba; i++)
	{
		aux = (i * dimens.columnas) + columna;
		resultado [aux] = matriz_comp [aux];
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
__global__ void eliminar_columna_cuda (unsigned long semilla,
				       int *resultado,
				       const int min,
				       const int max,
				       const Dim dimens,
				       int col_bomba)
{
	int fila = blockIdx.y * blockDim.y + threadIdx.y,
	    i,
	    rand_int,
	    aux;

	curandState estado;
	extern __shared__ int matriz_comp [];

	float rand_f;

	if ( (fila >= dimens.filas)
		|| ( (blockIdx.x * blockDim.x + threadIdx.x) != 0) )
	{
		return;
	}

	/* Copia la fila en la memoria compartida */
	for (i = 0; i <= col_bomba; i++)
	{
		aux = (fila * dimens.columnas) + i;
		matriz_comp [aux] = resultado [aux];
	}

	/* ---- A partir de aquí, trabaja con la memoria compartida ---- */

	/* Intercambia los elementos desde la fila actual hasta el principio */
	for (i = col_bomba; i > 0; i--)
	{
		aux = (fila * dimens.columnas) + i;
		matriz_comp [aux] = matriz_comp [aux - 1];
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
	matriz_comp [fila * dimens.columnas] = rand_int;

	/* Copia los datos de vuelta a la memoria global */
	for (i = 0; i <= col_bomba; i++)
	{
		aux = (fila * dimens.columnas) + i;
		resultado [aux] = matriz_comp [aux];
	}
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
	extern __shared__ int matriz_comp [];

	if ((fila >= dimens.filas)
		|| (columna >= dimens.columnas))
	{
		return;
	}

	if (comprobar_giro (fila, columna, dimens))
	{
		/* Copia el cuadrante en la memoria compartida (desenrrollamiento de
		un bucle 'for') */
		aux = (posY * dimens.columnas) + posX;
		matriz_comp [aux] = resultado [aux];
		matriz_comp [aux + 1] = resultado [aux + 1];
		matriz_comp [aux + 2] = resultado [aux + 2];

		aux = ( (posY + 1) * dimens.columnas) + posX;
		matriz_comp [aux] = resultado [aux];
		matriz_comp [aux + 1] = resultado [aux + 1];
		matriz_comp [aux + 2] = resultado [aux + 2];

		aux = ( (posY + 2) * dimens.columnas) + posX;
		matriz_comp [aux] = resultado [aux];
		matriz_comp [aux + 1] = resultado [aux + 1];
		matriz_comp [aux + 2] = resultado [aux + 2];
		/* ---- A partir de aquí, se usa la memoria compartida ---- */


		/* Se realizan los intercambios de manera manual */
		aux = matriz_comp [(posY * dimens.columnas) + posX];
		/* ---- */
		matriz_comp [(posY * dimens.columnas) + posX]
			= matriz_comp [( (posY + 2) * dimens.columnas) + posX];

		matriz_comp [( (posY + 2) * dimens.columnas) + posX]
			= matriz_comp [( (posY + 2) * dimens.columnas) + posX + 2];

		matriz_comp [( (posY + 2) * dimens.columnas) + posX + 2]
			= matriz_comp [(posY * dimens.columnas) + posX + 2];

		matriz_comp [(posY * dimens.columnas) + posX + 2] = aux;

		/* ---- */
		aux = matriz_comp [(posY * dimens.columnas) + posX + 1];

		matriz_comp [(posY * dimens.columnas) + posX + 1]
			= matriz_comp [( (posY + 1) * dimens.columnas) + posX];

		matriz_comp [( (posY + 1) * dimens.columnas) + posX]
			= matriz_comp [( (posY + 2) * dimens.columnas) + posX + 1];

		matriz_comp [( (posY + 2) * dimens.columnas) + posX + 1]
			= matriz_comp [( (posY + 1) * dimens.columnas) + posX + 2];

		matriz_comp [( (posY + 1) * dimens.columnas) + posX + 2] = aux;

		/* Copia el cuadrante de nuevo en memoria global (desenrrollamiento de
		un bucle 'for') */
		aux = (posY * dimens.columnas) + posX;
		resultado [aux] = matriz_comp [aux];
		resultado [aux + 1] = matriz_comp [aux + 1];
		resultado [aux + 2] = matriz_comp [aux + 2];

		aux = ( (posY + 1) * dimens.columnas) + posX;
		resultado [aux] = matriz_comp [aux];
		resultado [aux + 1] = matriz_comp [aux + 1];
		resultado [aux + 2] = matriz_comp [aux + 2];

		aux = ( (posY + 2) * dimens.columnas) + posX;
		resultado [aux] = matriz_comp [aux];
		resultado [aux + 1] = matriz_comp [aux + 1];
		resultado [aux + 2] = matriz_comp [aux + 2];
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
	    aux;
	extern __shared__ int mem_comp [];

	int *matriz_comp,
	    tam_matriz = dimens.filas * dimens.columnas,
	    *coinc_comp;


	if ( (fila >= dimens.filas)
		|| ( (blockIdx.x * blockDim.x+ threadIdx.x) != 0) )
	{
		return;
	}

	/* Obtiene los punteros a las diferentes zonas de la memoria compartida */
	matriz_comp = mem_comp;
	coinc_comp = &mem_comp [tam_matriz];

	/* Copia la fila en la memoria compartida */
	for (i = 0; i < dimens.columnas; i++)
	{
		aux = (fila * dimens.columnas) + i;
		matriz_comp [aux] = matriz [aux];
		coinc_comp [aux] = coincidencias [aux];
	}

	/* ---- A partir de aquí, trabaja con la memoria compartida ---- */
	aux = fila * dimens.columnas;

	/* Recorre la matriz marcando los elementos iguales consecutivos */
	for (i = 0; i < (dimens.columnas - 2) ; i++)
	{
		if ( (matriz_comp [aux + i] == matriz_comp [aux + i + 1])
			&& (matriz_comp [aux + i] == matriz_comp [aux + i + 2]) )
		{
			coinc_comp [aux + i] = COINCIDE;
			coinc_comp [aux + i + 1] = COINCIDE;
			coinc_comp [aux + i + 2] = COINCIDE;
		}
	}

	/* Copia de vuelta los resultados (sólo hay que copiar la matriz con
	las coincidencias) */
	for (i = 0; i < dimens.columnas; i++)
	{
		aux = (fila * dimens.columnas) + i;
		coincidencias [aux] = coinc_comp [aux];
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
	    i,
	    aux;
	extern __shared__ int mem_comp [];

	int *matriz_comp,
	    tam_matriz = dimens.filas * dimens.columnas,
	    *coinc_comp;

	if ( (columna >= dimens.columnas)
		|| ( (blockIdx.y * blockDim.y + threadIdx.y) != 0) )
	{
		return;
	}

	/* Obtiene los punteros a las diferentes zonas de la memoria compartida */
	matriz_comp = mem_comp;
	coinc_comp = &mem_comp [tam_matriz];

	/* Copia la fila en la memoria compartida */
	for (i = 0; i < dimens.filas; i++)
	{
		aux = (i * dimens.columnas) + columna;
		matriz_comp [aux] = matriz [aux];
		coinc_comp [aux] = coincidencias [aux];
	}

	/* ---- A partir de aquí, trabaja con la memoria compartida ---- */

	/* Recorre la matriz marcando los elementos iguales consecutivos */
	for (i = 0; i < (dimens.filas - 2) ; i++)
	{
		aux = (i * dimens.columnas);

		if ( (matriz_comp [aux + columna]
			== matriz_comp [( (i + 1) * dimens.columnas) + columna])
			&& (matriz_comp [aux + columna]
				== matriz_comp [( (i + 2) * dimens.columnas) + columna]) )
		{
			coinc_comp [aux + columna] = COINCIDE;
			coinc_comp [(i + 1) * dimens.columnas + columna] = COINCIDE;
			coinc_comp [(i + 2) * dimens.columnas + columna] = COINCIDE;
		}
	}

	/* Copia de vuelta los resultados (sólo hay que copiar la matriz con
	las coincidencias) */
	for (i = 0; i < dimens.filas; i++)
	{
		aux = (i * dimens.columnas) + columna;
		coincidencias [aux] = coinc_comp [aux];
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
	    rand_int,
	    aux;

	extern __shared__ int matriz_comp [];

	curandState estado;
	float rand_f;

	if ( (columna >= dimens.columnas)
		|| ( (blockIdx.y * blockDim.y + threadIdx.y) != 0) )
	{
		return;
	}

	/* Copia la columna en la memoria compartida */
	for (i = 0; i < dimens.filas; i++)
	{
		aux = (i * dimens.columnas) + columna;
		matriz_comp [aux] = resultado [aux];
	}

	/* ---- A partir de aquí, trabaja con la memoria compartida ---- */

	/* Recorre la columna hasta encontrar un elemento vacío */
	for (i = (dimens.filas - 1); i >= 0; i--)
	{
		aux = (i * dimens.columnas) + columna;
		elem = matriz_comp [aux];

		if (elem == DIAMANTE_VACIO)
		{
			/* Busca el primer elemento que haya por encima y lo baja */
			elem = buscar_lleno (matriz_comp, i, columna, dimens);

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

			matriz_comp [aux] = elem;
		}
	}

	/* Copia de vuelta los resultados */
	for (i = 0; i < dimens.filas; i++)
	{
		aux = (i * dimens.columnas) + columna;
		resultado [aux] = matriz_comp [aux];
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

	KERNEL_COMP (err, eliminar_fila_cuda,
		bloques, hilos,
		malla->dimens.columnas * (fila_bomba + 1) * sizeof matriz_d [0],
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

	KERNEL_COMP (err, eliminar_columna_cuda,
		bloques, hilos,
		malla->dimens.filas * (col_bomba + 1) * sizeof matriz_d [0],
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

	KERNEL_COMP (err, girar_matriz_cuda,
		bloques, hilos,
		tam * sizeof matriz_d [0],
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

	KERNEL_COMP (err, busar_coinc_cuda_fila,
		bloques, hilos,
		(2 * tam * sizeof matriz_d [0]),
		matriz_d, malla->dimens, coincidencias_d
	);

	dim_matr_hilos.filas = 1;
	dim_matr_hilos.columnas = malla->dimens.columnas;
	obtener_dim (&bloques, &hilos, dim_matr_hilos);

	KERNEL_COMP (err, busar_coinc_cuda_col,
		bloques, hilos,
		(2 * tam * sizeof matriz_d [0]),
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

	KERNEL_COMP (err, llenar_vacios_cuda,
		bloques, hilos,
		tam * sizeof matriz_d [0],
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



/* -------- */

/**
 * Busca las mejores jugadas por filas
 *
 * @param matriz
 *		Matriz con los valores actuales de los diamantes.
 *
 * @param dimens
 *		Estructura con las dimensiones de la matriz.
 *
 * @param mat2
 *		Matriz en la que se va a indicar los valores de las mejores 
 *		jugadas por posicion
 *
 * @param solh
 *		Matriz que devuleve las mejores jugdas(posicion, movimiento 
 *		y valor de cada jugada).
 */
__global__ void realizar_jugada_horizontal_cuda (int * mat1, Dim dimens, int * mat2,int * solh)
{
 
	int fila = blockIdx.x * blockDim.x + threadIdx.x;
	int aux = fila * dimens.columnas;
	   

	if ((fila >= dimens.filas)
		|| ( (blockIdx.y * blockDim.y + threadIdx.y) != 0) )
	{
		return;
	}

	for(int i = 0; i < dimens.filas * 4; i++){
		solh[i] = 0;
	}

	//Variable para recorrer la matriz
	int fin = 0;
	int cont = 0;
	int mov = 1;
	int ref = 0;

	//Vareables no definitivas para guardar el movimiento que se va a realizar
	
	int sen = 0;
	int posY = 0;
	int posX = 0;
	
	//Mejor movimiento horizontal
	
	int mh_sen = 0;  //Sentido del movimeinto
	int mh_posY = 0;
	int mh_posX = 0;
	

	for (int i = 0; i < dimens.columnas; i++)
	{
		ref = mat1[aux + i];

		for(int j = i; j < dimens.columnas; j ++)
		{
			if(fin == 0)
			{
				if(ref == mat1[aux + j])
				{
					//Mira si la posicion en la que esta es igual a la referencia
					cont ++;
				}
				else if((mov == 1)&&(fila > 0)&&(ref == mat1[aux - dimens.columnas + j]))
				{
					//Mira la posicion de arriba --> mover mat1[aux + j] arriba
					mov = 0;
					cont ++;

					sen = 3; posY = fila; posX = j;
				}
				else if((mov == 1)&&(fila < dimens.filas - 1)&&(ref == mat1[aux + dimens.columnas + j]))
				{
					//Mira la posicion de abajo --> mover mat1[aux + j] abajo
					mov = 0;
					cont ++;

					sen = 1; posY = fila;posX = j;
				}
				else if((mov == 1)&&((j + 1) < dimens.columnas)&&(ref == mat1[aux + j + 1]))
				{
					//Mirar la posicion de la derecha --> mover mat1[aux + j] derecha
					mov = 0;
					cont ++;

					sen = 0; posY = fila; posX = j;

					j++; //Pasa a comprobar la siguiente

				}
				else
				{
					fin = 1;
				}
			}
		}	
		//Mirar en las posiciones de la izquierda
		if ((mov == 1)&&(i > 0)&&(ref == mat1 [aux - dimens.columnas + i - 1]))
		{
			//Mirar la posicion por el lado de la izquierda arriba --> mover mat[aux + i - 1] arriba
			mov = 0;
			cont ++;

			sen = 3; posY = fila; posX = i - 1;
		}
		else if ((mov == 1)&&(i > 0)&&(ref == mat1 [aux + dimens.columnas + i -1]))
		{
			//Mirar la posicion por el lado de la izquierda abajo --> mover mat[aux + i - 1] abajo
			mov = 0;
			cont ++;

			sen = 1; posY = fila; posX = i - 1;
		}
		
		if(solh[fila * 4] <= cont){
			
			mh_sen = sen;
			if (mov == 1) sen = 4;
			mh_posY = posY;
			mh_posX = posX;
			
		}
		
		if((solh[fila * 4] == 0)||(solh [fila * 4] < cont))
		{
			solh[fila * 4] = cont;
			solh[(fila * 4) + 1] = mh_sen;
			solh[(fila * 4) + 2] = mh_posY;
			solh[(fila * 4) + 3] = mh_posX;
		}
		
		mat2[aux + i] = cont;
		
		//Reinicia valores
		mov = 1;
		fin = 0;
		cont = 0;
	}
}

/**
 * Busca las mejores jugadas por filas
 *
 * @return
 *		SUCCESS si todo ha salido correctamente.
 *		ERR_CUDA si hubo alg˙n error al obtener las caracterÌsticas del
 *	dispositivo.
 *		ERR_TAM si la matriz especificada sobrepasa las capacidades del
 *	dispositivo.
 */
int realizar_jugada_horizontal(Malla * malla,int * jugada)
{

	cudaError_t err;
	dim3 bloques, hilos;

	int tam = malla->dimens.columnas * malla->dimens.filas;
	
	int * matriz_d;
	int * mov_d;

	int * aux = (int *) malloc  (tam * sizeof aux [0]);

	//vector solh

	int * vec = (int *) malloc (malla->dimens.filas * sizeof (int) * 4);
	int * vec_d;

	cudaMalloc((void **) &vec_d,malla->dimens.filas * sizeof (int) * 4);

	for (int i = 0; i < malla->dimens.filas * 4; i++)
	{
		vec[i] = 0;
	}

	//Crea un hilo por columna
	Dim dim_matr_hilos;

	dim_matr_hilos.filas = malla->dimens.filas;
	dim_matr_hilos.columnas = 1;

	//Inicializa la matriz auxiliar 
	int idx;

	for (int i = 0; i < malla->dimens.filas; i++)
	{
		for (int j = 0; j < malla->dimens.columnas; j++)
		{
			idx = (i * malla->dimens.columnas) + j;
			aux[idx] = malla->matriz[idx].id;
		}
	}

	CUDA (err, cudaMalloc((void **)&matriz_d,tam * sizeof matriz_d[0]));

	CUDA (err, cudaMalloc((void **)&mov_d,tam * sizeof mov_d[0]));

	CUDA (err, cudaMemset(mov_d, NO_COINCIDE, tam * sizeof mov_d[0]));

	CUDA (err, cudaMemcpy(matriz_d, aux, tam * sizeof matriz_d [0],cudaMemcpyHostToDevice));

	cudaMemcpy (vec_d, &vec , sizeof (int) * malla->dimens.filas , cudaMemcpyHostToDevice);

	/* Llama al nucelo para comprar la matriz */
	obtener_dim (&bloques, &hilos, dim_matr_hilos);

	KERNEL (err, realizar_jugada_horizontal_cuda, bloques,hilos,matriz_d,malla->dimens,mov_d,vec_d);

	CUDA (err, cudaMemcpy(aux,mov_d, tam * sizeof aux [0], cudaMemcpyDeviceToHost));

	CUDA (err, cudaMemcpy(vec, vec_d, sizeof(int)*malla->dimens.filas *4,cudaMemcpyDeviceToHost));

	for (int i = 0; i < malla->dimens.filas; i++)
	{
		if(jugada[0]<vec[i * 4])
		{
			jugada[0] = vec[i * 4];
			jugada[1] = vec[i * 4 +1];
			jugada[2] = vec[i * 4 +2];
			jugada[3] = vec[i * 4 +3];
		}	
	}

	//copiar_matriz (aux, malla);

	CUDA (err, cudaFree(matriz_d));

	CUDA (err, cudaFree(mov_d));

	return SUCCESS;
}

/**
 * Busca las mejores jugadas por columnas
 *
 * @param matriz
 *		Matriz con los valores actuales de los diamantes.
 *
 * @param dimens
 *		Estructura con las dimensiones de la matriz.
 *
 * @param mat2
 *		Matriz en la que se va a indicar los valores de las mejores 
 *		jugadas por posicion
 *
 * @param solv
 *		Matriz que devuleve las mejores jugdas(posicion, movimiento 
 *		y valor de cada jugada).
 */
__global__ void realizar_jugada_vertical_cuda (int * mat1, Dim dimens, int * mat2, int * solv)
{

	int columna = blockIdx.y * blockDim.y + threadIdx.y;

	if ((columna >= dimens.columnas)
		|| ( (blockIdx.x * blockDim.x + threadIdx.x) != 0) )
	{
		return;
	}

	for(int i = 0;i < dimens.columnas * 4; i++){
		solv[i] = 0;
	}


	//Variable para recorrer la matriz
	int fin = 0;
	int cont = 0;
	int mov = 1;
	int ref = 0;

	//Vareables no definitivas para guardar el movimiento que se va a realizar
	
	int sen = 0;
	int posY = 0;
	int posX = 0;
	
	//Mejor movimiento horizontal
	
	int mh_sen = 0;  //Sentido del movimeinto
	int mh_posY = 0;
	int mh_posX = 0;
	

	for (int i = 0; i < dimens.filas; i++)
	{
		ref = mat1[dimens.columnas * i + columna];

		for(int j = i; j < dimens.filas; j ++)
		{
			if(fin == 0)
			{
				if(ref == mat1[dimens.columnas * j + columna])
				{
					//Mira si la posicion en la que esta es igual a la referencia
					cont ++;
				}
				
				else if((mov == 1)&&(columna > 0)&&(ref == mat1[dimens.columnas * j + columna - 1]))
				{
					//Mira la posicion de la izquierda --> mover mat1[dimens.columnas * j + columna - 1] izquierda
					mov = 0;
					cont ++;

					sen = 2; posY = j; posX = columna;
				}
				else if((mov == 1)&&(columna < dimens.columnas - 1)&&(ref == mat1[dimens.columnas * j + columna + 1]))
				{
					//Mira la posicion de la derecha --> mover mat1[dimens.columnas * j + columna + 1] derecha
					mov = 0;
					cont ++;

					sen = 0; posY = j;posX = columna;
				}
				
				else if((mov == 1)&&((j + 1) < dimens.filas)&&(ref == mat1[dimens.columnas * (j+1) + columna]))
				{
					//Mirar la posicion de abajo --> mover mat1[dimens.columnas * j + columna] abajo
					mov = 0;
					cont ++;

					sen = 1; posY = j; posX = columna;

					j++; //Pasa a comprobar la siguiente

				}
				else
				{
					fin = 1;
				}
			}
		}
		//Mirar en las posiciones de arriba
		if ((mov == 1)&&(columna > 0)&&(i>0)&&(ref == mat1 [dimens.columnas * (i- 1) + (columna - 1)]))
		{
			//Mirar la posicion por el lado de arriba izquierda --> mover mat[dimens.columnas * (i- 1) + (columna - 1)] izquierda
			mov = 0;
			cont ++;

			sen = 2; posY = i -1 ; posX =columna;
		}
		else if ((mov == 1)&&(i > 0)&&(columna + 1 < dimens.columnas)&&(ref == mat1 [dimens.columnas * (i - 1) + (columna + 1)]))
		{
			//Mirar la posicion por el lado de la arriba derecha --> mover mat[dimens.columnas * (i - 1) + (columna + 1)] derecha
			mov = 0;
			cont ++;

			sen = 0; posY = i - 1; posX = columna;
		}

		if(solv[columna * 4] <= cont){
	
			mh_sen = sen;
			if (mov == 1) sen = 4;
			mh_posY = posY;
			mh_posX = posX;
			
		}
		
		if((solv[columna * 4] == 0)||(solv[columna * 4] < cont))
		{
			solv[columna * 4] = cont;
			solv[(columna * 4) + 1] = mh_sen;
			solv[(columna * 4) + 2] = mh_posY;
			solv[(columna * 4) + 3] = mh_posX;	
		}
		
		mat2[dimens.columnas * i + columna] = cont;
		
		//Reinicias valores
		mov = 1;
		fin = 0;
		cont = 0;
	}

}

/**
 * Busca las mejores jugadas por columnas
 *
 * @return
 *		SUCCESS si todo ha salido correctamente.
 *		ERR_CUDA si hubo alg˙n error al obtener las caracterÌsticas del
 *	dispositivo.
 *		ERR_TAM si la matriz especificada sobrepasa las capacidades del
 *	dispositivo.
 */
int realizar_jugada_vertical(Malla * malla,int * jugada)
{

	cudaError_t err;
	dim3 bloques, hilos;

	int tam = malla->dimens.columnas * malla->dimens.filas;
	
	int * matriz_d;
	int * mov_d;

	int * aux = (int *) malloc  (tam * sizeof aux [0]);

	int * vec = (int *) malloc (malla->dimens.columnas * sizeof (int) * 4);
	int * vec_d;

	cudaMalloc((void **) &vec_d,malla->dimens.columnas * sizeof (int) * 4);

	for (int i = 0; i < malla->dimens.columnas * 4; i++)
	{
		vec[i] = 0;
	}

	//Crea un hilo por fila
	Dim dim_matr_hilos;

	dim_matr_hilos.filas = 1;
	dim_matr_hilos.columnas = malla->dimens.columnas;

	//Inicializa la matriz auxiliar 
	int idx;

	for (int i = 0; i < malla->dimens.filas; i++)
	{

		for (int j = 0; j < malla->dimens.columnas; j++)
		{
			idx = (i * malla->dimens.columnas) + j;
			aux[idx] = malla->matriz[idx].id;
		}
	}
	

	CUDA (err, cudaMalloc((void **)&matriz_d,tam * sizeof matriz_d[0]));

	CUDA (err, cudaMalloc((void **)&mov_d,tam * sizeof mov_d[0]));

	CUDA (err, cudaMemset(mov_d, NO_COINCIDE, tam * sizeof mov_d[0]));

	CUDA (err, cudaMemcpy(matriz_d, aux, tam * sizeof matriz_d [0],cudaMemcpyHostToDevice));

	cudaMemcpy( vec_d, &vec,sizeof (int) * malla->dimens.columnas, cudaMemcpyHostToDevice );

	obtener_dim (&bloques, &hilos, dim_matr_hilos);

	KERNEL (err, realizar_jugada_vertical_cuda, bloques,hilos,matriz_d,malla->dimens,mov_d,vec_d);

	CUDA (err, cudaMemcpy(aux,mov_d, tam * sizeof aux [0], cudaMemcpyDeviceToHost));
	
	CUDA (err, cudaMemcpy(vec, vec_d,sizeof (int) * malla->dimens.columnas * 4, cudaMemcpyDeviceToHost ));
	
	for (int i = 0; i < malla->dimens.columnas; i++)
	{
		if(jugada[0]<vec[i * 4])
		{
			jugada[0] = vec[i * 4];
			jugada[1] = vec[i * 4 +1];
			jugada[2] = vec[i * 4 +2];
			jugada[3] = vec[i * 4 +3];
		}	
	}

	CUDA (err, cudaFree(matriz_d));

	CUDA (err, cudaFree(mov_d));

	return SUCCESS;
}

int realizar_jugada(Malla * malla)
{
	int jugada_v[4];
	int jugada_h[4];

	int posY;
	int posX;
	int mov;

	realizar_jugada_vertical(malla,jugada_v);
	realizar_jugada_horizontal(malla,jugada_h);

	if(jugada_v[0]>jugada_h[0])
	{
		printf("Mejor jugada --> Mov(%d): %d PosY: %d PosX: %d\n",jugada_v[0],jugada_v[1],jugada_v[2],jugada_v[3]);
		posY = jugada_v[2];
		posX = jugada_v[3];
		mov = jugada_v[1];
	}
	else
	{
		printf("Mov(%d): %d PosY: %d PosX: %d\n",jugada_h[0],jugada_h[1],jugada_h[2],jugada_h[3]);
		posY = jugada_h[2];
		posX = jugada_h[3];
		mov = jugada_h[1];
	}
	mover_diamante(posY,posX,mov,* malla);
	return SUCCESS;
}
