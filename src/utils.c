#include "common.h"
#include "libutils.h"


/* ------------------ */
/* VARIABLES GLOBALES */
/* ------------------ */

/**
 * Nivel de detalle para los mensajes (para mostrar mensajes de depuración, por ejemplo).
 */
int nivel_detalle = 0;

/**
 * Nivel actual en el juego.
 */
int nivel = 1;

/**
 * Dimensiones de la matriz de juego.
 *
 * (por defecto, es de 4 x 5)
 */
Dim tam_matriz = {
		.filas = 4,
		.columnas = 5
	};

/**
 * Indica si el modo es automático (true) o manual (false).
 */
bool modo_auto = true;


/**
 * Procesa los argumentos pasados por línea de comandos
 *
 * @param
 *		Número de argumentos pasados (nº de elementos en argv).
 *
 * @param argv
 *		Array de cadenas con los argumentos.
 *
 *
 * @return
 *		-> ERR_ARGS si se ha especificado alguna opción no reconocida.
 *		-> SUCCESS si se han procesado los argumentos correctamente y se
 *			debe proseguir con la ejecución.
 *		-> SUCC_ARGS si se ha procesado un argumento y se debe terminar la
 *			ejecución (p.ej.: tras procesar '-h').
 */
int procesar_args (int argc, char *argv [])
{
	int res,
	    aux = 0;

	while ((res = getopt (argc, argv, "hman:f:c:v")) != -1)
	{
		switch (res)
		{
			case 'h':
				imprimir (DETALLE_LOG, "%s\n", MSG_AYUDA);
				return SUCC_ARGS;

			case 'm':
				modo_auto = false;
				break;

			case 'a':
				modo_auto = true;
				break;

			case 'n':
				aux = atoi (optarg);
				/* Si el nivel especificado es menor que o igual a 0,
				(atoi devuelve 0 si se introduce algo que no es un
				entero) se establece el nivel 1.
					Si se indica un nivel mayor que el máximo
				permitido, se establece el máximo nivel */
				nivel = (aux <= 0)?
					1
					: (aux > MAX_NV)? MAX_NV : aux;
				break;

			case 'f':
				aux = atoi (optarg);
				tam_matriz.filas = (aux <= 0)? 1 : aux;
				break;

			case 'c':
				aux = atoi (optarg);
				tam_matriz.columnas = (aux <= 0)? 1 : aux;
				break;

			case 'v':
				/* Aumenta el nivel de detalle */
				nivel_detalle ++;
				break;

			default:
				return ERR_ARGS;
		}
	}

	return SUCCESS;
}

/**
 * Imprime toda la información de las variables globales del juego.
 */
void imprimir_info ()
{
	imprimir (DETALLE_DEBUG,
		"------------------------------------------\n"
		"Valor de las variables globales del juego:\n"
			"\t--> Nivel de detalle: %i\n"
				"\t--> Dimensiones del juego:\n"
				"\t\tFilas: %i\n"
				"\t\tColumnas: %i\n"
			"\t--> Nivel: %i\n"
			"\t--> Modo: %s\n"
		"------------------------------------------\n",
		nivel_detalle,
		tam_matriz.filas,
		tam_matriz.columnas,
		nivel,
		(modo_auto)? "automático" : "manual");
}

/**
 * Cambia el valor de los parámetros del juego.
 *
 * @param nuevos_params
 * 		Estructura de tipo Malla (definida en 'common.h') con los nuevos nivel
 * 	y dimensiones del tablero de juego.
 */
void cambiar_params (Malla nuevos_params)
{
	nivel = nuevos_params.nivel;
	tam_matriz.filas = nuevos_params.dimens.filas;
	tam_matriz.columnas = nuevos_params.dimens.columnas;
}


/**
 * Devuelve una estructura Malla con los valores especificados (nivel y dimensiones),
 * pero sin ninguna memoria reservada para la matriz.
 *
 *
 * @return
 * 		Una nueva instancia de tipo Malla, con los valores especificados por
 * 	línea de comandos.
 */
Malla ver_params ()
{
	Malla malla = {

		.dimens = tam_matriz,
		.nivel = nivel,
		.matriz = 0
	};

	return malla;
}


/**
 * Permite guardar la malla en el fichero especificado.
 *
 * @param malla
 * 		Estructura con toda la información del juego actual (nivel, dimensiones
 * 	de la matriz y el contenido de la matriz).
 *
 * @param nombre_fichero
 * 		Nombre del fichero en el que se deben guardar los datos. Si ya existe se
 * 	sobrescribirá; si no, se creará.
 *
 *
 * @return
 * 		SUCCESS si los datos se han guardado correctamente.
 * 		ERR_ARCHIVO si no se pudo abrir o cerrar correctamente el archivo.
 */
int guardar (Malla malla, const char *nombre_fichero)
{
	FILE *fichero = fopen (nombre_fichero, "w+");
	int i,
	    j,
	    filas = malla.dimens.filas,
	    columnas = malla.dimens.columnas;

	if (fichero == NULL)
	{
		imprimir (DETALLE_LOG,
			 "Error al abrir el archivo '%s'\n",
			  nombre_fichero);
		return ERR_ARCHIVO;
	}

	/* Vuelca el contenido en el archivo en el siguiente orden (el mismo en el que
	están declarados en 'common.h'):
		Nivel
		Dimensiones:
			Filas
			Columnas
		Matriz
	*/
	fprintf (fichero,
		 "-> Nivel: %i\n"
		 "-> Dimensiones:\n"
		 "\t--> Filas: %i\n"
		 "\t--> Columnas: %i\n"
		 "Matriz:\n",
		 malla.nivel,
		 filas,
		 columnas);

	/* Recorre a matriz guardando su contenido por filas */
	for (i = 0; i < filas; i++)
	{
		for (j = 0; j < columnas; j++)
		{
			fprintf (fichero, "%5i ", malla.matriz [(i * columnas) + j].id);
		}
		fprintf (fichero, "\n");
	}

	if (fclose (fichero) != 0)
	{
		imprimir (DETALLE_LOG,
			 "Error al cerrar el archivo '%s'\n",
			  nombre_fichero);
		return ERR_ARCHIVO;
	}

	return SUCCESS;
}

/**
 * Carga desde el fichero especificado el juego guardado.
 *
 * @param malla
 * 		Estructura en la que se va a cargar la información del juego.
 *
 * @param nombre_fichero
 * 		Nombre del fichero que contiene la información del juego.
 *
 *
 * @return
 * 		SUCCESS si el archivo se cargó correctamente.
 * 		ERR_ARCHIVO si hubo algún error al abrir o cerrar el fichero.
 */
int cargar (Malla *malla, const char *nombre_fichero)
{
	FILE *fichero = fopen (nombre_fichero, "r");
	int i,
	    j,
	    pos,
	    aux;

	if (fichero == NULL)
	{
		printf ("Error al abrir el archivo.\n");
		return ERR_ARCHIVO;
	}

	/* Obtiene el contenido del archivo en el siguiente orden (el mismo en el que
	están declarados en 'common.h'):
		Nivel
		Dimensiones:
			Filas
			Columnas
		Matriz
	*/
	fscanf (fichero,
		 "-> Nivel: %i\n"
		 "-> Dimensiones:\n"
		 "\t--> Filas: %i\n"
		 "\t--> Columnas: %i\n"
		 "Matriz:\n",
		 &malla->nivel,
		 &malla->dimens.filas,
		 &malla->dimens.columnas);

	/* Reserva memoria para la matriz */
	malla->matriz = (Diamante *) malloc (malla->dimens.filas
						* malla->dimens.columnas
						* sizeof malla->matriz[0]);
	if (malla->matriz == NULL)
	{
		imprimir (DETALLE_LOG, "Error al reservar memoria para la matriz.\n");
		return ERR_MEM;
	}

	/* Rellena la matriz con los valores del fichero */
	for (i = 0; i < malla->dimens.filas; i++)
	{
		for (j = 0; j < malla->dimens.columnas; j++)
		{
			pos = (i * malla->dimens.columnas) + j;
			fscanf (fichero, "%5d ", &aux);

			malla->matriz [pos] = crear_diamante (aux);
		}
		fscanf (fichero, "\n");
	}

	/* Cierra el archivo */
	if (fclose (fichero) != 0)
	{
		imprimir (DETALLE_LOG,
			 "Error al cerrar el archivo '%s'\n",
			  nombre_fichero);
		return ERR_ARCHIVO;
	}


	return SUCCESS;
}

/**
 * Reserva la memoria necesaria para el tablero de juego
 *
 * @param malla
 * 		Estructura de tipo Malla (definida en 'common.h') con las dimensiones de
 * 	la matriz y su contenido.
 *
 *
 * @return
 * 		SUCCESS si todo ha salido correctamente
 * 		ERR_MEM si hubo algún error al intentar reservar la memoria.
 */
int reservar_mem (Malla *malla)
{
	int filas = malla->dimens.filas,
	    columnas = malla->dimens.columnas,
	    i,
	    j,
	    aux;

	malla->matriz = (Diamante *) malloc (filas * columnas * sizeof malla->matriz[0]);

	if (malla->matriz == NULL)
	{
		imprimir (DETALLE_LOG,
			  "Error al intentar reservar la memoria para la matriz\n");
		return ERR_MEM;
	}

	/* Inicializa la matriz para poner todas las casillas vacías */
	for (i = 0; i < filas; i++)
	{
		for (j = 0; j< columnas; j++)
		{
			//malla->matriz [(i * columnas) + j].id = DIAMANTE_VACIO;
			aux = (i * columnas) + j;
			malla->matriz [aux] = crear_diamante (DIAMANTE_VACIO);
		}
	}

	return SUCCESS;
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
 * 		SUCCESS si todo ha salido correctamente
 */
int rellenar (Malla *malla)
{
	int filas = malla->dimens.filas,
	    columnas = malla->dimens.columnas,
	    i,
	    j;
	Diamante diamante;
	int max = DIAMANTE_VACIO;

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
		imprimir (DETALLE_DEBUG, "Error al intentar reservar la memoria para la matriz\n");
		return ERR_MEM;
	}

	srand (time (NULL));
	/* Asigna valores aleatorios al diamante */
	for (i = 0; i < filas; i++)
	{
		for (j = 0; j < columnas; j++)
		{
			diamante = crear_diamante ((rand () % max) + 1);
			malla->matriz [(i * columnas) + j] = diamante;
		}
	}

	return SUCCESS;
}


/**
 * Imprime por pantalla el contenido de la matriz.
 *
 * @param malla
 * 		Estructura de tipo Malla (definida en 'common.h') con las dimensiones de
 * 	la matriz y su contenido.
 */
void mostrar_malla (Malla malla)
{
	int i,
	    j,
	    filas = malla.dimens.filas,
	    columnas = malla.dimens.columnas;

	for (i = 0; i < filas; i++)
	{
		for (j = 0; j < columnas; j++)
		{
			imprimir (DETALLE_LOG,
				  "%s ",
				  malla.matriz [(i * columnas) + j].img);
		}
		imprimir (DETALLE_LOG, "\n");
	}
}

/**
 * Crea un diamante del tipo especificado.
 *
 * @param num
 * 		Número del tipo de diamante a crear.
 *
 * @return
 * 		Una nueva instancia de diamante.
 */
Diamante crear_diamante (int num)
{
	/* Si se pasa por parametro num = 0, seria como eliminar un dimante */
	Diamante d;

	d.id = num;

	switch (num)
	{
		case 0: d.img = NOR "#" RST; break;
		case 1: d.img = AZU "1" RST; break;
		case 2: d.img = ROJ "2" RST; break;
		case 3: d.img = CYN "3" RST; break;
		case 4: d.img = VER "4" RST; break;
		case 5: d.img = ROS "5" RST; break;
		case 6: d.img = AMA "6" RST; break;
		case 7: d.img = NOR "7" RST; break;
		case 8: d.img = BLA "8" RST; break;
	}

	return d;
}


/**
 * Imprime por pantalla la cadena pasada como argumento sólo si el nivel de detalle es
 * el especificado (o mayor).
 *
 * @param detalle
 * 		Nivel de detalle mínimo para imprimir el mensaje
 *
 * @param cadena
 * 		Cadena con formato para imprimir
 *
 * @param ...
 * 		Argumentos para el formato de la cadena
 */
void imprimir (int detalle, const char *cadena, ...)
{
	va_list args;
	va_start (args, cadena);

	if (nivel_detalle >= detalle)
	{
		vprintf (cadena, args);
	}

	va_end (args);
}
