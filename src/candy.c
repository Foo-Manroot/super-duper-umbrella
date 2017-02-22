#include "candy.h"

#define FIL ver_params ().dimens.filas
#define COL ver_params ().dimens.columnas

void mostrar_malla(Diamante malla[FIL][COL]){
	for (int i = 0; i < FIL; ++i)
	{
		for (int j = 0; j < COL; ++j)
		{
			printf("%d ",malla[i][j].id);
		}
		printf("\n");
	}
	
}

Diamante generar_diamante(){
	Diamante d;

	d.id = (rand() % 9) + 1;

	return d;
}

Diamante cambiar_diamante(int num){
	/*Si se pasa por parametro num = 0, seria como eliminar un dimante*/
	Diamante d;

	d.id = num;
	
	return d;
}

void iniciar_malla(Diamante malla[FIL][COL]){
	
	for (int i = 0; i < FIL; ++i)
	{
		for (int j = 0; j < COL; ++j)
		{
			malla[i][j].id = 0; 
		}
	}	
}

void rellenar_malla(Diamante malla[FIL][COL]){
	for (int i = 0; i < FIL; ++i)
	{
			for (int j = 0; j < COL; ++j)
			{
			malla[i][j]= generar_diamante(); 
			}
	}	
}

int es_valido(int posY, int posX, int mov, Diamante malla[FIL][COL]){
	/*Si devuelve 0 no hay error*/
	int b = 1;
	switch(mov){
		case 0:
			/*Mover hacia abajo*/
			if(posY < (FIL-1)) b = 0;
			break;
		case 1:
			/*Mover hacia arriba*/
			if(posY > 0) b = 0;
			break;
		case 2:
			/*Mover hacia izquierda*/
			if(posX > 0) b = 0;
			break;
		case 3:
			/*Mover hacia derecha*/
			if(posX < (COL-1)) b = 0;
			break;
	}
	return b;
}

void mover_diamante(int posY, int posX, int mov, Diamante malla[FIL][COL]){
	Diamante aux;

	if (es_valido(posY,posX,mov,malla) == 0){

		switch(mov){
		case 0:
			/*Mover hacia abajo*/
			aux = malla[posY + 1][posX];
			malla[posY + 1][posX] = malla[posY][posX];
			malla[posY][posX] = aux;

			break;
		case 1:
			/*Mover hacia arriba*/
			aux = malla[posY - 1][posX];
			malla[posY - 1][posX] = malla[posY][posX];
			malla[posY][posX] = aux;
			break;
		case 2:
			/*Mover hacia izquierda*/
			aux = malla[posY][posX - 1];
			malla[posY][posX - 1] = malla[posY][posX];
			malla[posY][posX] = aux;
			break;
		case 3:
			/*Mover hacia derecha*/
			aux = malla[posY ][posX + 1];
			malla[posY][posX + 1] = malla[posY][posX];
			malla[posY][posX] = aux;
			break;
		}
	}else printf("Error: movimeinto no valido\n");
}

int son_iguales(Diamante d1,Diamante d2){
	//Compara si dos diamantes son del mismo tipo
	int b = 1;
	if ((d1.id != 0)&&(d1.id == d2.id)) b = 0;
	return b;
}

int buscar_coincidencias(int posY, int posX, int sen,Diamante malla[FIL][COL]){

	int posY_sig;
	int posX_sig;



	if(es_valido(posY,posX,sen,malla) == 0){
		//printf ("[%d][%d], %d \n",posY,posX,sen);
		switch(sen){
			case 0: posY_sig = posY + 1; posX_sig = posX; break;
			case 1:	posY_sig = posY - 1; posX_sig = posX; break;
			case 2: posY_sig = posY; posX_sig = posX - 1; break;
			case 3: posY_sig = posY; posX_sig = posX + 1; break;
		}

		if(son_iguales(malla[posY][posX],malla[posY_sig][posX_sig]) == 0){
			return (buscar_coincidencias(posY_sig,posX_sig,sen,malla) + 1); //Suma una coincidencia
		}
	}
	return 0;
}

void eliminar_coincidencias(int posY,int posX,int sen,Diamante malla[FIL][COL]){

	int posY_sig;
	int posX_sig;

	if(es_valido(posY,posX,sen,malla) == 0){

		switch(sen){
			case 0: posY_sig = posY + 1; posX_sig = posX; break;
			case 1:	posY_sig = posY - 1; posX_sig = posX; break;
			case 2: posY_sig = posY; posX_sig = posX - 1; break;
			case 3: posY_sig = posY; posX_sig = posX + 1; break;
		}
		
		if(son_iguales(malla[posY][posX],malla[posY_sig][posX_sig]) == 0){
			eliminar_coincidencias(posY_sig,posX_sig,sen,malla);
			malla[posY_sig][posX_sig] = cambiar_diamante(0);
		}
	}

}

void eliminar_coincidencias_eje(int posY, int posX,int eje,Diamante malla[FIL][COL]){
	switch(eje){
		case 0: //Eje vertical
			eliminar_coincidencias(posY,posX,0,malla);
			eliminar_coincidencias(posY,posX,1,malla);
			malla[posY][posX] = cambiar_diamante(0);

			break;
		case 1://Eje horizontal

			eliminar_coincidencias(posY,posX,2,malla);
			eliminar_coincidencias(posY,posX,3,malla);
			malla[posY][posX] = cambiar_diamante(0);

			break;
	}

}

void tratar_coincidencias(int posY, int posX, Diamante malla[FIL][COL]) {

	/**
	* int sen{sentido}
	* Arriba = 1
	* Abajo = 0
	* Izquierda = 2
	* Derecha = 3
	*/

	if(malla[posY][posX].id != 0){ //Si la posicion no es un hueco
		/*Busqueda horizontal*/
		if((buscar_coincidencias(posY,posX,2,malla) + buscar_coincidencias(posY,posX,3,malla) + 1) >= 3){
			eliminar_coincidencias_eje(posY,posX,1,malla);//Eje horizontal 1
		}
		/*Busqueda vertical*/
		if((buscar_coincidencias(posY,posX,0,malla) + buscar_coincidencias(posY,posX,1,malla) + 1) >= 3){
			eliminar_coincidencias_eje(posY,posX,0,malla);//Eje vertical 0
		}	
	}
}

void recorrer_malla_coincidencias(Diamante malla[FIL][COL]){
	for (int i = 0; i < FIL; ++i){
	
		for (int j = 0; j < COL; ++j)
		{
			tratar_coincidencias(i,j,malla);	
		}
	}
}

void tratar_huecos(int posY, int posX, Diamante malla[FIL][COL]){
	
	if((posY < (FIL - 1))&&(malla[posY + 1][posX].id == 0)){
		mover_diamante(posY,posX,0,malla);
		if (posY == 0){
			malla[posY][posX] = generar_diamante();
		}else{
			tratar_huecos(posY - 1,posX,malla);
		}
		
	}

}

void recorrer_malla_huecos(Diamante malla[FIL][COL]){
	for (int i = 0; i < FIL; ++i){
	
		for (int j = 0; j < COL; ++j)
		{
			tratar_huecos(i,j,malla);	
		}
	}
}
void eliminar_fila(int fila,Diamante malla[FIL][COL]){
	if (fila < FIL){
		for (int i = 0; i < FIL; ++i)
		{
			malla[fila][i] = cambiar_diamante(0);
		}
	}
	recorrer_malla_huecos(malla);

}

void reordenar_tablero(int posY,int posX,Diamante malla[FIL][COL]){
	if((posX < (COL - 1))&&(malla[posY][posX + 1].id == 0)){
		mover_diamante(posY,posX,3,malla);
		if (posX == 0){
			malla[posY][posX] = generar_diamante();
		}else{
			reordenar_tablero(posY,posX - 1,malla);
		}
		
	}

}


void recorrer_malla_reorden(Diamante malla[FIL][COL]){
	for (int i = 0; i < FIL; ++i){
	
		for (int j = 0; j < COL; ++j)
		{
			reordenar_tablero(i,j,malla);	
		}
	}
}

void eliminar_columna(int columna,Diamante malla[FIL][COL]){
	if (columna < COL){
		for (int i = 0; i < COL; ++i)
		{
			malla[i][columna] = cambiar_diamante(0);
		}
		recorrer_malla_reorden(malla);
	}
}

/**
 * Función principal
 */
int main (int argc, char *argv[])
{

	Diamante  malla[FIL][COL];

	/**
	 * Procesa los argumentos por línea de comandos
 	 */
	int ret = procesar_args (argc, argv);

	if (ret == SUCC_ARGS)
	{
		/* Sale del juego (se ha especificado alguna opción que lo
		 indica, como '-h') */
		return SUCCESS;
	}


/* ---- */
Malla test = {

	.dimens = ver_params ().dimens,
	.nivel = ver_params ().nivel
};

reservar_mem (&test);
rellenar (&test);

imprimir_malla (test);
return 0;
/* ---- */

	iniciar_malla(malla);
	rellenar_malla(malla);
	
//        system("clear");
	for (int i = 0; i < 10; ++i)
	{

		mostrar_malla(malla);
		/*
		int posY,posX,mov;		
		printf("PosY: ");
        scanf("%d", &(posY));
        printf("PosX: ");
        scanf("%d", &(posX));
        printf("Mov{abajo = 0, arriba = 1, izquierda = 2, derecha = 3}: ");
        scanf("%d", &(mov));
        */
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

        }

        system("clear");
        //Refresh
		//mover_diamante(posY,posX,mov,malla);
		recorrer_malla_coincidencias(malla);
		recorrer_malla_huecos(malla);


	}

	return SUCCESS;
}


/* ---------------- */
/* IMPLEMENTACIONES */
/* ---------------- */

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
	    j;

	malla->matriz = malloc (filas * columnas * sizeof malla->matriz);

	if (malla->matriz == NULL)
	{
		printf ("Error al intentar reservar la memoria para la matriz\n");
		return ERR_MEM;
	}

	/* Inicializa la matriz para poner todas las casillas vacías */
	for (i = 0; i < filas; i++)
	{
		for (j = 0; j< columnas; j++)
		{
			malla->matriz [(i * columnas) + j].id = DIAMANTE_VACIO;
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
		printf ("Error al intentar reservar la memoria para la matriz\n");
		return ERR_MEM;
	}

	srand (time (NULL));
	/* Asigna valores aleatorios al diamante */
	for (i = 0; i < filas; i++)
	{
		for (j = 0; j< columnas; j++)
		{
			diamante.id = (rand () % max) + 1;
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
void imprimir_malla (Malla malla)
{
	int i,
	    j,
	    filas = malla.dimens.filas,
	    columnas = malla.dimens.columnas;

	for (i = 0; i < filas; i++)
	{
		for (j = 0; j < columnas; j++)
		{
			printf ("%5i ", malla.matriz [(i * columnas) + filas].id);
		}
		printf ("\n");
	}
}
