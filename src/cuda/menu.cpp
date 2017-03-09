#include "include/menu.h"

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
	bool fin = false;

	while (!fin)
	{
		mostrar_malla (malla);
		/* Imprime el menú y permite elegir opciones */
		imprimir (DETALLE_LOG, MSG_MENU);

		/* Pide la opción seleccionada (entre el 0 y el 6) */
		selecc = pedir_opcion (0, 6);

		/*
		Opciones disponibles:
			0.- Salir
			1.- Mover diamante
			2.- Bomba
			3.- Guardar partida
			4.- Cargar partida
			5.- Cambiar nivel
			6.- Jugada automática
		*/
		switch (selecc)
		{
			case 0:
				fin = true;
				break;
			case 1:
				mover (malla);
				break;
			case 2:
				bomba (malla);
				break;
			case 3:
				guardar_partida (malla);
				break;
			case 4:
				cargar_partida (&malla);
				break;
			case 5:
				cambiar_nivel (&malla);
				break;
			case 6:
				imprimir (DETALLE_LOG, "Opción aún no implementada.\n");
				break;
			default:
				imprimir (DETALLE_LOG, "Opción no reconocida.\n");
		}
		recorrer_malla_coincidencias (malla);
		recorrer_malla_huecos (malla);
	}
}

/**
 * Pide los datos necesarios para realizar un movimiento.
 *
 * @param malla
 * 		Estructura con la matriz en la que se van a mover los elementos.
 */
void mover (Malla malla)
{
	int posY = 0,
	    posX = 0,
	    mov = 0;

	/* Pide los datos */
	imprimir (DETALLE_LOG, "Posición del elemento a mover:\n");
	/* Fila */
	imprimir (DETALLE_LOG,
		 "\tFila (entre 0 y %i): ",
		 (malla.dimens.filas - 1));

	posY = pedir_opcion (0, (malla.dimens.filas - 1));
	/* Columna */
	imprimir (DETALLE_LOG,
		 "\tColumna (entre 0 y %i): ",
		 (malla.dimens.columnas - 1));

	posX = pedir_opcion (0, (malla.dimens.columnas - 1));
	/* Movimiento */
	imprimir (DETALLE_LOG, "Movimiento. Opciones disponibles:\n"
				"\t%i -> derecha\n"
				"\t%i -> abajo\n"
				"\t%i -> izquierda\n"
				"\t%i -> arriba\n"
				"Introduzca el movimiento seleccionado: ",
				MOV_DER,
				MOV_ABAJO,
				MOV_IZQ,
				MOV_ARRIBA);
	mov = pedir_opcion (MOV_DER, MOV_ARRIBA);

	/* Realiza el movimiento y comprueba si hay elementos alineados */
	mover_diamante (posY, posX, mov, malla);
	recorrer_malla_coincidencias (malla);
}


/**
 * Pide los datos necesarios para usar una bomba.
 *
 *  @param malla
 * 		Estructura con la matriz en la que se van a mover los elementos.
 */
void bomba (Malla malla)
{
	int bomba = 1,
	    fila = 0,
	    columna = 0;

	imprimir (DETALLE_LOG, "Bombas disponibles:\n"
				"\t1 -> Eliminar fila\n"
				"\t2 -> Eliminar columna\n"
				"\t3 -> Girar en grupos de 3x3\n"
				"Introduzca la bomba seleccionada: ");
	bomba = pedir_opcion (1, 3);

	switch (bomba)
	{
		case 1:
			imprimir (DETALLE_LOG,
				  "Fila (entre 0 y %i): ",
				  (malla.dimens.filas - 1));

			fila = pedir_opcion (0, (malla.dimens.filas - 1));
			/* Llama a la función que utiliza CUDA */
			bomba_fila (fila, &malla);
			break;

		case 2: 
			imprimir (DETALLE_LOG,
				  "Columna (entre 0 y %i): ",
				  (malla.dimens.columnas - 1));

			columna = pedir_opcion (0, (malla.dimens.columnas - 1));

			eliminar_columna (columna, malla);
			break;

		case 3: 
			recorrer_malla_giro (malla);
			break;
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
 *
 *
 * @return
 * 		El valor introducido por teclado.
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
			ret_val = -1;
			imprimir (DETALLE_LOG,
				 "Sólo se admiten números entre %i y %i.\n"
				 "Introduzca de nuevo un valor: ",
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

	/* Guarda la partida */
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

/**
 * Muestra las opciones para cargar un fichero con una partida.
 *
 * @param antigua
 * 		Malla con la información de la partida actual. Si el fichero se carga
 * 	correctamente, se sustituye por ella. Si no, se deja como estaba.
 */
void cargar_partida (Malla *antigua)
{
	Malla nueva;
	char fichero [100];

	/* Pide el nombre del fichero */
	imprimir (DETALLE_LOG, "Introduzca el nombre del archivo a cargar: ");

	fgets (fichero, sizeof fichero, stdin);
	/* Sustituye el salto de línea final */
	fichero [strcspn (fichero, "\r\n")] = 0;

	if (cargar (&nueva, fichero) == SUCCESS)
	{
		imprimir (DETALLE_LOG,
			  "Partida cargada desde '%s' correctamente.\n",
			  fichero);

		antigua->matriz = nueva.matriz;
		antigua->dimens = nueva.dimens;
		antigua->nivel = nueva.nivel;

		cambiar_params (nueva);
	}
	else
	{
		imprimir (DETALLE_LOG,
			  "Error al cargar la partida del archivo '%s'.\n",
			  fichero);
	}
}


/**
 * Cambia el nivel del juego.
 *
 * @param malla
 * 		Estructura (definida en 'common.h') en la que se almacena la información
 * 	del juego y que va a ser cambiada.
 */
void cambiar_nivel (Malla *malla)
{
	int nuevo_nivel = malla->nivel;

	/* Pide un número para cambiar el nivel */
	imprimir (DETALLE_LOG, "Introduzca el nuevo nivel (entre 1 y %i): ", MAX_NV);
	nuevo_nivel = pedir_opcion (1, MAX_NV);

	imprimir (DETALLE_LOG, "Nuevo nivel: %i.\n", nuevo_nivel);

	malla->nivel = nuevo_nivel;

	cambiar_params (*malla);
}
