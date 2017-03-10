#include "include/candy.h"


int main (int argc, char *argv[])
{
	/* Comprueba los argumentos */
	if (procesar_args (argc, argv) == SUCC_ARGS)
	{
		return SUCCESS;
	}

//	Malla malla = {
//		.dimens = ver_params ().dimens,
//		.nivel = ver_params ().nivel
//	};
	/* El compilador es una mierda, así que hay que inicializar a mano */
	Malla malla;
	malla.dimens = ver_params ().dimens;
	malla.nivel = ver_params ().nivel;

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
	struct timeval delta;
	gettimeofday(&delta, NULL);

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

	srand(delta.tv_usec);
	d = crear_diamante ((rand()% max) + 1);

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
 * 		Movimiento a realizar (definidos en 'candy.h'):
 * 			MOV_DER:	0 -> derecha
 * 			MOV_ABAJO:	1 -> abajo
 * 			MOV_IZQ:	2 -> izquierda
 * 			MOV_ARRIBA:	3 -> arriba
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
		case MOV_DER:
			/*Mover hacia derecha*/
			aux = matriz [(posY * cols) + posX + 1];
			matriz [(posY * cols) + posX + 1] = matriz [posY * cols + posX];
			matriz [(posY * cols) + posX] = aux;
			break;
		case MOV_ABAJO:
			/*Mover hacia abajo*/
			aux = matriz [((posY + 1) * cols) + posX];
			matriz [((posY + 1) * cols) + posX] = matriz [(posY * cols) + posX];
			matriz [(posY * cols) + posX] = aux;
			break;
		case MOV_IZQ:
			/*Mover hacia izquierda*/
			aux = matriz [(posY * cols) + posX - 1];
			matriz [(posY * cols) + posX - 1] = matriz [(posY * cols) + posX];
			matriz [(posY * cols) +  posX] = aux;
			break;
		case MOV_ARRIBA:
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

