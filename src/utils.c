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
				printf ("%s\n", MSG_AYUDA);
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
	if (nivel_detalle >= 1)
	{
		printf ("------------------------------------------\n"
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
		printf ("Error al abrir el archivo '%s'\n", nombre_fichero);
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
		printf ("Error al cerrar el archivo '%s'\n", nombre_fichero);
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
	    j;

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
	malla->matriz = malloc (malla->dimens.filas
				* malla->dimens.columnas
				* sizeof malla->matriz);
	if (malla->matriz == NULL)
	{
		printf ("Error al reservar memoria para la matriz.\n");
		return ERR_MEM;
	}

	/* Rellena la matriz con los valores del fichero */
	for (i = 0; i < malla->dimens.filas; i++)
	{
		for (j = 0; j < malla->dimens.columnas; j++)
		{
			fscanf (fichero,
				"%5i ",
				&malla->matriz [(i * malla->dimens.columnas) + j].id);
		}
		fscanf (fichero, "\n");
	}

	/* Cierra el archivo */
	if (fclose (fichero) != 0)
	{
		printf ("Error al cerrar el archivo '%s'\n", nombre_fichero);
		return ERR_ARCHIVO;
	}


	return SUCCESS;
}
