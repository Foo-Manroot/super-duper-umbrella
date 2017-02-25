#include "menu.h"

/**
 * Función principal para el menú del juego. Esta función se encarga de llamar al resto
 * en bucle.
 *
 * @param malla
 * 		Estructura con la información del juego.
 */
void menu (Malla malla)
{
	int selecc = 0;

	mostrar_malla (malla);

	while (true)
	{
		/* Imprime el menú y permite elegir opciones */
		imprimir (DETALLE_LOG, MSG_MENU);

		/* Pide la opción seleccionada (entre el 1 y el 5) */
		selecc = pedir_opcion (1, 5);

		/*
		Opciones disponibles:
			1.- Mover diamante
			2.- Bomba
			3.- Guardar partida
			4.- Cargar partida
			5.- Cambiar nivel
		*/
		switch (selecc)
		{
			case 1:
				break;
			case 2:
				break;
			case 3:
				guardar_partida (malla);
				break;
			case 4:
				break;
			case 5:
				break;
			default:
				imprimir(DETALLE_LOG, "Opción no reconocida.\n");
		}
	}
}


/**
 * Pide una opción por teclado y muestra un mensaje de error hasta que se ha introducido
 * un número válido en el rango especificado.
 *
 * @param min
 * 		Rango inferior del grupo de números admitidos (inclusivo).
 *
 * @param max
 * 		Rango superior del grupo de números admitidos (inclusivo).
 */
int pedir_opcion (int min, int max)
{
	int selecc = -1,
	    ret_val = -1;
	char entrada [50];

	do
	{
		/* Obtiene la entrada por teclado y busca un número. Si no lo encuentra,
		descarta la entrada hasta que se introduzca un número. */
		fgets (entrada, sizeof entrada, stdin);
		ret_val = sscanf (entrada, "%d", &selecc);

		if (ret_val != 1)
		{
			imprimir (DETALLE_LOG,
				  "Entrada incorrecta. Introduzca el número"
				  " de la opción seleccionada: ");
		}
		else if ((selecc < min)
			|| (selecc > max))
		{
			imprimir (DETALLE_LOG,
				 "Sólo se admiten números entre %i y %i\n",
				  min, max);
		}

	} while (ret_val != 1);

	return selecc;
}

/**
 * Muestra las opciones para guardar la partida actual.
 *
 * @param malla
 * 		Estructura de tipo Malla (definida en 'common.h') con la información de
 * 	la partida.
 */
void guardar_partida (Malla malla)
{
	char fichero [100];

	/* Pide el nombre del fichero */
	imprimir (DETALLE_LOG, "Introduzca el nombre del archivo en el que"
				" desea guardar los datos: ");

	fgets (fichero, sizeof fichero, stdin);

	/* Sustituye el salto de línea final */
	fichero [strcspn (fichero, "\r\n")] = 0;


	if (guardar (malla, fichero) == SUCCESS)
	{
		imprimir (DETALLE_LOG,
			  "Partida guardada en el archivo '%s'.\n",
			  fichero);
	}
	else
	{
		imprimir (DETALLE_LOG,
			  "Error al guardar la partida en '%s'.\n",
			  fichero);
	}
}
