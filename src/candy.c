#include "candy.h"


int main (int argc, char *argv[])
{
	/* Comprueba los argumentos */
	if (procesar_args (argc, argv) == SUCC_ARGS)
	{
		return SUCCESS;
	}

	Malla malla = {
		.dimens = ver_params ().dimens,
		.nivel = ver_params ().nivel
	};

	/* Reserva memoria para la matriz y la rellena */
	reservar_mem (&malla);
	rellenar (&malla);

	menu (malla);

	return SUCCESS;
/*
	for (int i = 0; i < 10; ++i)
	{
		mostrar_malla(malla);
		
		int posY,posX,mov;		
		printf("PosY: ");
        scanf("%d", &(posY));
        printf("PosX: ");
        scanf("%d", &(posX));
        printf("Mov{abajo = 1, arriba = 3, izquierda = 2, derecha = 0}: ");
        scanf("%d", &(mov));
*/

        /*
        int bomba;
        printf("Bomba: ");
        scanf("%d", &(bomba));

        int fila = 0;
        int columna = 0;

        switch(bomba){
        	case 1:
        		
        		printf("Fila: ");
        		scanf("%d", &(fila));
        		eliminar_fila(fila,malla);
        		break;
        	case 2: 
        		
        		printf("Columna: ");
        		scanf("%d", &(columna));
        		eliminar_columna(columna,malla);
        		break;
        	case 3: 
        		recorrer_malla_giro(malla);
        		break;

        }
		*/
/*        system("clear");

        //Refresh
		mover_diamante(posY,posX,mov,malla);
		recorrer_malla_coincidencias(malla);

		//
		system("clear");
		mostrar_malla(malla);

		scanf("%d", &(mov));

		//
		system("clear");
		recorrer_malla_huecos(malla);


	}

	return 0;
*/
}

/* ---------------- */
/* IMPLEMENTACIONES */
/* ---------------- */

/**
 * Crea un diamante nuevo con un identificador entre 1 y DIAMANTE_MAX (definido en
 * 'common.h'), según el nivel especificado como argumento por línea de comandos.
 *
 *
 * @return
 * 		Una nueva instancia de diamante ya inicializada con un id aleatorio.
 */
Diamante generar_diamante()
{
	Diamante d;
	int max = DIAMANTE_VACIO;

	switch (ver_params ().nivel)
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

	/* Genera un id aleatorio */
	srand (time (NULL));
	d = crear_diamante ((rand () % max) + 1);

	return d;
}
/*
 * es_valido()
 * Devuelve 1 si el movimineto puede realizarse
 * 
 * @param mov 
 *		Indica el sentido en el que se raliza el movimietno
 */
int es_valido(int posY, int posX, int mov, Malla malla)
{
	/*Si devuelve 0 hay error*/
	int b = 0,
	    cols = malla.dimens.columnas,
	    filas = malla.dimens.filas;

	switch(mov)
	{
		case 0:
			/*Mover hacia derecha*/
			if(posX < (cols-1)) b = 1;
			break;
		case 1:
			/*Mover hacia abajo*/
			if(posY < (filas-1)) b = 1;
			break;
		case 2:
			/*Mover hacia izquierda*/
			if(posX > 0) b = 1;
			break;
		case 3:
			/*Mover hacia arriba*/
			if(posY > 0) b = 1;
			break;

	}
	return b;
}


/**
 * Realiza un movimiento en la matriz.
 *
 * @param posY
 * 		Fila del elemento a mover (entre 0 y n-1).
 *
 * @param PosX
 * 		Columna del elemento a mover (entre 0 y n-1).
 *
 * @param mov
 * 		Movimiento a realizar:
 * 			0 -> derecha
 * 			1 -> abajo
 * 			2 -> izquierda
 * 			3 -> arriba
 *
 * @param malla
 * 		Estructura (definida en 'common.h') con los datos de la matriz de juego.
 */
void mover_diamante(int posY, int posX, int mov, Malla malla)
{
	Diamante aux;
	int cols = malla.dimens.columnas;
	Diamante *matriz = malla.matriz;

	if (es_valido(posY,posX,mov,malla) == 1)
	{

		switch(mov)
		{
		case 0:
			/*Mover hacia derecha*/
			aux = matriz [(posY * cols) + posX + 1];
			matriz [(posY * cols) + posX + 1] = matriz [posY * cols + posX];
			matriz [(posY * cols) + posX] = aux;
			break;
		case 1:
			/*Mover hacia abajo*/
			aux = matriz [((posY + 1) * cols) + posX];
			matriz [((posY + 1) * cols) + posX] = matriz [(posY * cols) + posX];
			matriz [(posY * cols) + posX] = aux;
			break;
		case 2:
			/*Mover hacia izquierda*/
			aux = matriz [(posY * cols) + posX - 1];
			matriz [(posY * cols) + posX - 1] = matriz [(posY * cols) + posX];
			matriz [(posY * cols) +  posX] = aux;
			break;
		case 3:
			/*Mover hacia arriba*/
			aux = matriz [((posY - 1) * cols) + posX];
			matriz [((posY - 1) * cols) + posX] = matriz [(posY * cols) + posX];
			matriz [(posY * cols) + posX] = aux;
			break;
		}
	}
	else
	{
		imprimir (DETALLE_LOG, "Error: movimiento no válido\n");
	}
}


int son_iguales(Diamante d1,Diamante d2)
{
	//Compara si dos diamantes son del mismo tipo
	int b = 0;
	if ((d1.id != 0)&&(d1.id == d2.id)) b = 1;
	return b;
}


/**
 * Devuelve el numero de conicidencias del elemnto en la matriz en un sentido
 *
 * @param posY
 * 		Fila del elemento (entre 0 y n-1).
 *
 * @param PosX
 * 		Columna del elemento (entre 0 y n-1).
 *
 * @param sen
 * 		Sentido en el que se realiza la busqueda:
 * 			0 -> derecha
 * 			1 -> abajo
 * 			2 -> izquierda
 * 			3 -> arriba
 *
 * @param malla
 * 		Estructura (definida en 'common.h') con los datos de la matriz de juego.
 */
int buscar_coincidencias(int posY, int posX, int sen,Malla malla)
{
	int posY_sig;
	int posX_sig;
	int cols = malla.dimens.columnas;
	Diamante *matriz = malla.matriz;

	if(es_valido(posY,posX,sen,malla) == 1)
	{
		//printf ("[%d][%d], %d \n",posY,posX,sen);
		switch(sen)
		{
			case 0: posY_sig = posY; posX_sig = posX + 1; break;
			case 1:	posY_sig = posY + 1; posX_sig = posX; break;
			case 2: posY_sig = posY; posX_sig = posX - 1; break;
			case 3: posY_sig = posY - 1; posX_sig = posX; break;
		}

		if(son_iguales(matriz [(posY * cols) + posX],
				matriz [(posY_sig * cols) + posX_sig]) == 1)
		{
			return (buscar_coincidencias(posY_sig,posX_sig,sen,malla) + 1); //Suma una coincidencia
		}
	}
	return 0;
}


/**
 * Elimina todos los elementos en una direccion que sean iguales al elemnto indicado
 *
 * @param posY
 * 		Fila del elemento (entre 0 y n-1).
 *
 * @param PosX
 * 		Columna del elemento (entre 0 y n-1).
 *
 * @param sen
 * 		Sentido en el que se esta borrando:
 * 			0 -> derecha
 * 			1 -> abajo
 * 			2 -> izquierda
 * 			3 -> arriba
 *
 * @param malla
 * 		Estructura (definida en 'common.h') con los datos de la matriz de juego.
 */
void eliminar_coincidencias(int posY,int posX,int sen,Malla malla)
{
	int posY_sig;
	int posX_sig;
	int cols = malla.dimens.columnas;
	Diamante *matriz = malla.matriz;

	if(es_valido(posY,posX,sen,malla) == 1){

		switch(sen){
			case 0: posY_sig = posY; posX_sig = posX + 1; break;
			case 1:	posY_sig = posY + 1; posX_sig = posX; break;
			case 2: posY_sig = posY; posX_sig = posX - 1; break;
			case 3: posY_sig = posY - 1; posX_sig = posX; break;
		}

		if(son_iguales(matriz [(posY * cols) + posX],
				matriz [(posY_sig * cols) + posX_sig]) == 1)
		{
			eliminar_coincidencias(posY_sig,posX_sig,sen,malla);
			matriz [(posY_sig * cols) + posX_sig] = crear_diamante(0);
		}
	}

}
/**
 * Divide entre los dos ejes la eliminación de elemntos 
 *
 * @param posY
 * 		Fila del elemento(entre 0 y n-1).
 *
 * @param PosX
 * 		Columna del elemento(entre 0 y n-1).
 *
 * @param mov
 * 		Eje sobre el que se va a eliminar:
 * 			0 -> vertical
 * 			1 -> horizontal
 * 			
 *
 * @param malla
 * 		Estructura (definida en 'common.h') con los datos de la matriz de juego.
 */
void eliminar_coincidencias_eje(int posY, int posX,int eje,Malla malla)
{
	Diamante *matriz = malla.matriz;
	switch(eje)
	{
		case 0: //Eje vertical
			eliminar_coincidencias(posY,posX,1,malla);
			eliminar_coincidencias(posY,posX,3,malla);
			matriz [(posY * malla.dimens.columnas) + posX] = crear_diamante(0);

			break;
		case 1://Eje horizontal

			eliminar_coincidencias(posY,posX,2,malla);
			eliminar_coincidencias(posY,posX,0,malla);
			matriz [(posY * malla.dimens.columnas) + posX] = crear_diamante(0);

			break;
	}

}

/**
 * Busca si hay 3 o mas coincidencias de un elemento en un mismo sentido .
 * Si las encuentra llama a la funcion eliminar_conicidencias_eje()
 *
 * @param posY
 * 		Fila del elemento(entre 0 y n-1).
 *
 * @param PosX
 * 		Columna del elemento(entre 0 y n-1).
 *
 * @param malla
 * 		Estructura (definida en 'common.h') con los datos de la matriz de juego.
 */
void tratar_coincidencias(int posY, int posX, Malla malla)
{
	Diamante *matriz = malla.matriz;

	if(matriz [(posY * malla.dimens.columnas) + posX].id != 0)
	{ //Si la posicion no es un hueco
		/*Busqueda horizontal*/
		if((buscar_coincidencias(posY,posX,2,malla) + buscar_coincidencias(posY,posX,0,malla) + 1) >= 3)
		{
			eliminar_coincidencias_eje(posY,posX,1,malla);//Eje horizontal 1
		}
		/*Busqueda vertical*/
		if((buscar_coincidencias(posY,posX,3,malla) + buscar_coincidencias(posY,posX,1,malla) + 1) >= 3)
		{
			eliminar_coincidencias_eje(posY,posX,0,malla);//Eje vertical 0
		}	
	}
}


void recorrer_malla_coincidencias (Malla malla)
{
	for (int i = 0; i < malla.dimens.filas; ++i)
	{
		for (int j = 0; j < malla.dimens.columnas; ++j)
		{
			tratar_coincidencias (i,j,malla);
		}
	}
}

/**
 * Rellena los huecos que se dejan cuando se eliminan los diamantes
 *
 * @param posY
 * 		Fila del elemento(entre 0 y n-1).
 *
 * @param PosX
 * 		Columna del elemento(entre 0 y n-1).
 *
 * @param malla
 * 		Estructura (definida en 'common.h') con los datos de la matriz de juego.
 */
void tratar_huecos(int posY, int posX, Malla malla)
{
	int cols = malla.dimens.columnas;
	Diamante *matriz = malla.matriz;

	if((posY == 0)&&(matriz [(posY * cols) + posX].id == 0)){

		matriz [(posY * cols) + posX] = generar_diamante();


		if(matriz [((posY + 1) * cols) + posX].id  == 0){
			mover_diamante(posY,posX,1,malla);	
			tratar_huecos(posY,posX,malla);
		}
	}
	else if((matriz [(posY * cols) + posX].id == 0)&&(matriz [((posY - 1) * cols) + posX].id != 0)) {

		mover_diamante(posY,posX,3,malla);
		tratar_huecos(posY -1,posX,malla);

	}
}


void recorrer_malla_huecos (Malla malla)
{
	for (int i = 0; i < malla.dimens.filas; ++i){
	
		for (int j = 0; j < malla.dimens.columnas; ++j)
		{
			tratar_huecos(i,j,malla);	
		}
	}
}

void eliminar_fila (int fila, Malla malla)
{
	int filas_malla = malla.dimens.filas;
	Diamante *matriz = malla.matriz;

	if (fila < filas_malla)
	{
		for (int i = 0; i < filas_malla; ++i)
		{
			matriz [(fila * malla.dimens.columnas) + i] = crear_diamante(0);
		}
	}
	recorrer_malla_huecos(malla);
}

/**
 * Reordenar tablero es auxiliar no tiene mucha importancia, no borrar 
 */
void reordenar_tablero (int posY,int posX,Malla malla)
{
	int cols = malla.dimens.columnas;
	Diamante *matriz = malla.matriz;

	if((posX < (cols - 1))&&(matriz [(posY * cols) + posX + 1].id == 0))
	{
		mover_diamante(posY,posX,0,malla);
		if (posX == 0)
		{
			matriz [(posY * cols) + posX] = generar_diamante();
		}
		else
		{
			reordenar_tablero(posY,posX - 1,malla);
		}
		
	}

}

void recorrer_malla_reorden (Malla malla)
{
	int filas = malla.dimens.filas,
	    cols = malla.dimens.columnas;

	for (int i = 0; i < filas; ++i)
	{	
		for (int j = 0; j < cols; ++j)
		{
			reordenar_tablero (i,j,malla);	
		}
	}
}


void eliminar_columna(int columna,Malla malla)
{
	int cols = malla.dimens.columnas;
	Diamante *matriz = malla.matriz;

	if (columna < cols)
	{
		for (int i = 0; i < cols; ++i)
		{
			matriz [(i * cols) + columna] = crear_diamante(0);
		}
		recorrer_malla_reorden(malla);
	}
}



int es_posible_giro (int posY, int posX, Malla malla)
{
	int res = 0,
	    filas = malla.dimens.filas,
	    cols = malla.dimens.columnas;

	/* Comprueba los límites del eje de giro */
	if ( ((posY + 1) >= filas)
		|| ((posX + 1) >= cols) )
	{
		return 0;
	}

	if(((posY - 1) == 0)
		|| ((posY - 1) % 3) == 0)
	{
		/* Posición correcta para el eje Y */
		if(((posX - 1) == 0)
			|| ((posX - 1) % 3) == 0)
		{
			/* Posición correcta para el eje X */
			res = 1;
		}

	}

	return res;
}

/**
 * La bomba III, 
 */
void girar_matriz(int ejeY, int ejeX, Malla malla)
{
	if (es_posible_giro (ejeY, ejeX, malla) == 1)
	{

		int posY = ejeY - 1;
		int posX = ejeX -1;
		
		Diamante aux;

		aux = malla.matriz [(posY * malla.dimens.columnas) + posX];
		mover_diamante(posY + 1,posX,3,malla);
		mover_diamante(posY + 2,posX,3,malla);
		mover_diamante(posY + 2,posX + 1,2,malla);
		mover_diamante(posY + 2,posX + 2,2,malla);
		mover_diamante(posY + 1,posX + 2,1,malla);
		mover_diamante(posY,posX + 2,1,malla);
		mover_diamante(posY,posX + 1,0,malla);
		malla.matriz [(posY *  malla.dimens.columnas) + posX + 1] = aux;
	}
}

void recorrer_malla_giro (Malla malla)
{
	int i,
	    j;

	for (i = 0; i < ver_params ().dimens.filas; ++i)
	{
		for (j = 0; j < ver_params ().dimens.columnas; ++j)
		{
			girar_matriz (i,j,malla);
		}
	}

}
