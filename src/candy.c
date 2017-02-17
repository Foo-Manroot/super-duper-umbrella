#include "candy.h"

#define FIL 5
#define COL 5

/* Estructura Diamante */
typedef struct {
	
	//id = {1,2,3,4,5,6,7,8} indican el tipo de diamante
	//id = {0} indica que no hay fiamante en ese hueco
	int id;
   
  } Diamante;

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

int es_valido_mov(int posY, int posX, int mov, Diamante malla[FIL][COL]){
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

	if (es_valido_mov(posY,posX,mov,malla) == 0){

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

void buscar_coincidencias(Diamante malla[FIL][COL]){
	for (int i = 0; i < FIL; ++i){
	
		for (int j = 0; j < COL; ++j)
		{
			Diamante d_cen = malla[i][j];
			Diamante d_der = malla[i][j + 1];
			Diamante d_izq = malla[i][j - 1];

			if((son_iguales(d_cen,d_der) == 0)&&
			(son_iguales(d_cen,d_izq) == 0)) {
			/*Se eliminan los diamantes cercanos que sean iguales*/

			malla[i][j] = cambiar_diamante(0);
			malla[i][j + 1] = cambiar_diamante(0);
			malla[i][j - 1] = cambiar_diamante(0);
			}	
		}
	}
}

/*void buscar_horizontal(int posY, int posX, Diamante malla[FIL][COL]){
	//int b = 1;

	Diamante d_cen = malla[posY][posX];
	Diamante d_der = malla[posY][posX + 1];
	Diamante d_izq = malla[posY][posX - 1];

	if((son_iguales(d_cen,d_der) == 0)&&
		(son_iguales(d_cen,d_izq) == 0)) {
		//Se eliminan los diamantes cercanos que sean iguales

		malla[posY][posX] = cambiar_diamante(0);
		malla[posY][posX + 1] = cambiar_diamante(0);
		malla[posY][posX - 0] = cambiar_diamante(0);



	}
	//return b;
}*/
int buscar_vertical(int posX, int posY, Diamante malla[FIL][COL]){
	int b = 1;

	return b;
}

int main(){

	Diamante  malla[FIL][COL];

	iniciar_malla(malla);
	rellenar_malla(malla);
	
	for (int i = 0; i < 10; ++i)
	{
		mostrar_malla(malla);
		int posY,posX,mov;		
		printf("PosY: ");
		scanf("%d", &(posY));
		printf("PosX: ");
		scanf("%d", &(posX));
		printf("Mov{abajo = 0, arriba = 1, izquierda = 2, derecha = 3}: ");
		scanf("%d", &(mov));

		
		
		system("clear");
		//Refresh
		mover_diamante(posY,posX,mov,malla);
		buscar_coincidencias(malla);


	}

	return 0;
}
