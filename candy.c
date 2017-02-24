#include<stdio.h>
#include<stdlib.h>

/*Colores*/

#define NOR  "\x1B[0m" //Color normal
#define ROJ  "\x1B[31m"
#define VER  "\x1B[32m"
#define AMA  "\x1B[33m"
#define AZU  "\x1B[34m"
#define ROS  "\x1B[35m"
#define CYN  "\x1B[36m"
#define BLA  "\x1B[37m"
#define RST "\x1B[0m" //RESET

/*Constates*/

#define FIL 3
#define COL 3


/* Estructura Diamante */
typedef struct {
    
    //id = {1,2,3,4,5,6,7,8} indican el tipo de diamante
    //id = {0} indica que no hay fiamante en ese hueco
    int id;
    char * img;
   
  } Diamante;

/*Muestra la matriz de diamantes*/
void mostrar_malla(Diamante malla[FIL][COL]){
	for (int i = 0; i < FIL; ++i)
	{
		for (int j = 0; j < COL; ++j)
		{
			printf("%s ",malla[i][j].img);
		}
		printf("\n");
	}
	
}

/*Cambia un diamante por otro*/
Diamante cambiar_diamante(int num){
	/*Si se pasa por parametro num = 0, seria como eliminar un dimante*/
	Diamante d;

	d.id = num;

	switch(num){
		case 0: d.img = NOR"#"RST; break;
		case 1: d.img = AZU"1"RST; break;
		case 2: d.img = ROJ"2"RST; break;
		case 3: d.img = CYN"3"RST; break;
		case 4: d.img = VER"4"RST; break;
		case 5: d.img = ROS"5"RST; break;
		case 6: d.img = AMA"6"RST; break;
		case 7: d.img = NOR"7"RST; break;
		case 8: d.img = BLA"8"RST; break;
	}


	
	return d;
}


/*Crea un diamante con identificador de 1 a 8*/
Diamante generar_diamante(){
	Diamante d;
	int num;

	num = (rand() % 9) + 1;
	d = cambiar_diamante(num);

	return d;
}

/*Inicia la malla vacia*/
void iniciar_malla(Diamante malla[FIL][COL]){
	
	for (int i = 0; i < FIL; ++i)
	{
		for (int j = 0; j < COL; ++j)
		{
			malla[i][j] = cambiar_diamante(0); 
		}
	}	
}

/*Inicia la malla con elementos aleatorios*/
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
	/*Si devuelve 0 hay error*/
	int b = 0;
	switch(mov){
		case 0:
			/*Mover hacia derecha*/
			if(posX < (COL-1)) b = 1;
			break;
		case 1:
			/*Mover hacia abajo*/
			if(posY < (FIL-1)) b = 1;
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

void mover_diamante(int posY, int posX, int mov, Diamante malla[FIL][COL]){
	Diamante aux;

	if (es_valido(posY,posX,mov,malla) == 1){

		switch(mov){
		case 0:
			/*Mover hacia derecha*/
			aux = malla[posY][posX + 1];
			malla[posY][posX + 1] = malla[posY][posX];
			malla[posY][posX] = aux;
			break;
		case 1:

			/*Mover hacia abajo*/
			aux = malla[posY + 1][posX];
			malla[posY + 1][posX] = malla[posY][posX];
			malla[posY][posX] = aux;
			break;
		case 2:
			/*Mover hacia izquierda*/
			aux = malla[posY][posX - 1];
			malla[posY][posX - 1] = malla[posY][posX];
			malla[posY][posX] = aux;
			break;
		case 3:
			/*Mover hacia arriba*/
			aux = malla[posY - 1][posX];
			malla[posY - 1][posX] = malla[posY][posX];
			malla[posY][posX] = aux;
			break;
		}
	}else printf("Error: movimeinto no valido [%d][%d], sentido: %d \n",posY,posX,mov);
}

int son_iguales(Diamante d1,Diamante d2){
	//Compara si dos diamantes son del mismo tipo
	int b = 0;
	if ((d1.id != 0)&&(d1.id == d2.id)) b = 1;
	return b;
}

int buscar_coincidencias(int posY, int posX, int sen,Diamante malla[FIL][COL]){

	int posY_sig;
	int posX_sig;



	if(es_valido(posY,posX,sen,malla) == 1){
		//printf ("[%d][%d], %d \n",posY,posX,sen);
		switch(sen){
			case 0: posY_sig = posY; posX_sig = posX + 1; break;
			case 1:	posY_sig = posY + 1; posX_sig = posX; break;
			case 2: posY_sig = posY; posX_sig = posX - 1; break;
			case 3: posY_sig = posY - 1; posX_sig = posX; break;
		}

		if(son_iguales(malla[posY][posX],malla[posY_sig][posX_sig]) == 1){
			return (buscar_coincidencias(posY_sig,posX_sig,sen,malla) + 1); //Suma una coincidencia
		}
	}
	return 0;
}

void eliminar_coincidencias(int posY,int posX,int sen,Diamante malla[FIL][COL]){

	int posY_sig;
	int posX_sig;

	if(es_valido(posY,posX,sen,malla) == 1){

		switch(sen){
			case 0: posY_sig = posY; posX_sig = posX + 1; break;
			case 1:	posY_sig = posY + 1; posX_sig = posX; break;
			case 2: posY_sig = posY; posX_sig = posX - 1; break;
			case 3: posY_sig = posY - 1; posX_sig = posX; break;
		}
		
		if(son_iguales(malla[posY][posX],malla[posY_sig][posX_sig]) == 1){
			eliminar_coincidencias(posY_sig,posX_sig,sen,malla);
			malla[posY_sig][posX_sig] = cambiar_diamante(0);
		}
	}

}

void eliminar_coincidencias_eje(int posY, int posX,int eje,Diamante malla[FIL][COL]){
	switch(eje){
		case 0: //Eje vertical
			eliminar_coincidencias(posY,posX,1,malla);
			eliminar_coincidencias(posY,posX,3,malla);
			malla[posY][posX] = cambiar_diamante(0);

			break;
		case 1://Eje horizontal

			eliminar_coincidencias(posY,posX,2,malla);
			eliminar_coincidencias(posY,posX,0,malla);
			malla[posY][posX] = cambiar_diamante(0);

			break;
	}

}



void tratar_coincidencias(int posY, int posX, Diamante malla[FIL][COL]) {


	if(malla[posY][posX].id != 0){ //Si la posicion no es un hueco
		/*Busqueda horizontal*/
		if((buscar_coincidencias(posY,posX,2,malla) + buscar_coincidencias(posY,posX,0,malla) + 1) >= 3){
			eliminar_coincidencias_eje(posY,posX,1,malla);//Eje horizontal 1
		}
		/*Busqueda vertical*/
		if((buscar_coincidencias(posY,posX,3,malla) + buscar_coincidencias(posY,posX,1,malla) + 1) >= 3){
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
	
	
	if((posY == 0)&&(malla[posY][posX].id == 0)){

		malla[posY][posX] = generar_diamante();


		if(malla[posY + 1][posX].id  == 0){			
			mover_diamante(posY,posX,1,malla);	
			tratar_huecos(posY,posX,malla);
		}
	}
	else if((malla[posY][posX].id == 0)&&(malla[posY - 1][posX].id != 0)){


		mover_diamante(posY,posX,3,malla);	
		tratar_huecos(posY -1,posX,malla);

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
		mover_diamante(posY,posX,0,malla);
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

int es_posible_giro(int posY, int posX, Diamante malla[FIL][COL]){
	int res = 0;

	if(((posY - 1)== 0)||((posY - 1) % 3) == 0){
		/* Posición correcta para el eje Y */
		if(((posX - 1)== 0)||((posX - 1)% 3) == 0){
			/* Posición correcta para el eje X */
			res = 1;
		}

	}
	return res;
}

void girar_matriz(int ejeY, int ejeX, Diamante malla[FIL][COL]){

	

	if(es_posible_giro(ejeY,ejeX,malla) == 1){

		int posY = ejeY - 1;
		int posX = ejeX -1;
		
		Diamante aux;

		aux = malla[posY][posX];
		mover_diamante(posY + 1,posX,3,malla);
		mover_diamante(posY + 2,posX,3,malla);
		mover_diamante(posY + 2,posX + 1,2,malla);
		mover_diamante(posY + 2,posX + 2,2,malla);
		mover_diamante(posY + 1,posX + 2,1,malla);
		mover_diamante(posY,posX + 2,1,malla);
		mover_diamante(posY,posX + 1,0,malla);
		malla[posY][posX+1] = aux;


		

	}
}

void recorrer_malla_giro(Diamante malla[FIL][COL]){
	for (int i = 0; i < FIL; ++i)
	{
		for (int j = 0; j < COL; ++j)

		{
			girar_matriz(i,j,malla);

			
		}
	}

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
        printf("Mov{abajo = 1, arriba = 3, izquierda = 2, derecha = 0}: ");
        scanf("%d", &(mov));
        
		iniciar_malla(malla);
        
        /*int bomba;
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

        }*/
		
        system("clear");

        //Refresh
		//mover_diamante(posY,posX,mov,malla);
		//recorrer_malla_coincidencias(malla);

		//system("clear");
		recorrer_malla_huecos(malla);


	}

	return 0;
}