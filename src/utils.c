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
dim_t tam_matriz = {
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
				(atoi devuelve 0 si se introduce algo que no es un entero)
				se establece el nivel 1.
					Si se indica un nivel mayor que el máximo permitido,
				se establece el máximo nivel */
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
