#include "include/candy.h"

/* ---- VARIABLES GLOBALES PARA OPENGL ---- */

/**
 * Variable para poder dibujar la matriz con OpenGL
 */
Malla malla;

/**
 * Tamaño del lado de las celdas de la matriz.
 */
float lado = 0.5;

/**
 * Espacio que se debe dejar entre las celdas.
 */
float espacio = 0.05;

/**
 * Dimensiones de la ventana y la pantalla.
 */
Dim ventana,
    pantalla;

/**
 * Casilla seleccionada actualmente
 */
Dim casilla_sel;

/**
 * Bandera que indica que se quiere lanzar una bomba.
 */
bool bomba_act = false;

/**
 * Bomba seleccionada (-1 indica que no hay ninguna)
 */
int sel_bomba = -1;

/**
 * Cuando está a 'true', significa que ha habido algún cambio.
 */
bool cambio = false;

/**
 * Contador para controlar los FPS (de una manera un poco chapucera).
 */
int contador = 0;


/* ----*------------------------------*---- */

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
	malla.dimens = ver_params ().dimens;
	malla.nivel = ver_params ().nivel;

	/* Reserva memoria para la matriz y la rellena */
	reservar_mem (&malla);
	rellenar (&malla);

	iniciar_opengl (argc, argv);

	return SUCCESS;
}

/* ---------------- */
/* IMPLEMENTACIONES */
/* ---------------- */

/**
 * Se encarga de todo lo necesario para iniciar la ventana con OpenGL.
 */
void iniciar_opengl (int argc, char *argv [])
{
	glutInit (&argc, argv);

	pantalla.filas = glutGet (GLUT_SCREEN_HEIGHT);
	pantalla.columnas = glutGet (GLUT_SCREEN_WIDTH);

	/* Inicializa la ventana con los datos necesarios (posición, tamaño...) */
	glutInitWindowSize (glutGet (GLUT_SCREEN_WIDTH), glutGet (GLUT_SCREEN_HEIGHT));

	glutInitWindowPosition (-1, -1);
	glutInitDisplayMode (GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
 
	glutCreateWindow ("Candy");

	/* Establece las funciones manejadoras */
	glutDisplayFunc (render);
	glutReshapeFunc (manejador_redim);
	glutIdleFunc (render);

	glutKeyboardFunc (manejador_teclas);
	glutMouseFunc (manejador_raton);

	/* Deshabilita la repetición de eventos al presionar una tecla */
	glutSetKeyRepeat (GLUT_KEY_REPEAT_OFF);

	ventana.filas = glutGet (GLUT_WINDOW_HEIGHT);
	ventana.columnas = glutGet (GLUT_WINDOW_WIDTH);

	imprimir (DETALLE_EXTRA,
		  "Dimensiones de la pantalla: %d x %d.\n"
		  "Dimensiones de la ventana: %d x %d.\n",
		  pantalla.filas, pantalla.columnas,
		  ventana.filas, ventana.columnas);

	casilla_sel.filas = -1;
	casilla_sel.columnas = -1;

	/* Inicia el bucle principal */
	glutMainLoop ();

	/* Restablece los valores necesarios */
	glutSetKeyRepeat (GLUT_KEY_REPEAT_DEFAULT);
}

/**
 * Dibuja una casilla de la matriz de diamantes.
 *
 * @param x
 * 		Posición X de la esquina superior izquierda de la celda.
 *
 * @param y
 * 		Posición Y de la esquina superior izquierda de la celda.
 *
 * @param posX
 * 		Posición X del elemento de la matriz a dibujar.
 *
 * @param posY
 * 		Posición Y del elemento de la matriz a dibujar.
 */
void dibujar_casilla (float x, float y, int posX, int posY)
{
	/* Obtiene el color del diamante */
	switch (malla.matriz [(posY * malla.dimens.columnas) + posX].id)
	{
		case 1: glColor3ub (0, 0, 255); break;
		case 2: glColor3ub (255, 0, 0); break;
		case 3: glColor3ub (0, 200, 200); break;
		case 4: glColor3ub (0, 255, 0); break;
		case 5: glColor3ub (200, 0, 200); break;
		case 6: glColor3ub (200, 200,0); break;
		case 7: glColor3ub (100, 100, 100); break;
		case 8: glColor3ub (255, 255, 255); break;
		default: glColor3ub (255, 255, 255);
	}

	/* Dibuja el cuadrado */
	glBegin (GL_POLYGON);
		glVertex2f (x	    , y	      );
		glVertex2f (x + lado, y	      );
		glVertex2f (x + lado, y + lado);
		glVertex2f (x 	    , y	+ lado);
	glEnd ();
}


/**
 * Dibuja los elementos en la pantalla.
 */
void render (void)
{
	int i,
	    j;

	float x = ( ((float) malla.dimens.columnas) * ((float) (lado + espacio)) ),
	      y = ( ((float) malla.dimens.filas) * ((float) (lado + espacio)) );

	if (contador < UMBRAL_FPS)
	{
		contador++;
		return;
	}
	contador = 0;

	/* Limpia el buffer */
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity ();

	/* Establece la posición de la cámara */
	gluLookAt (0.0f, 0.0f, 5.0f,
		   0.0f, 0.0f, 0.0f,
		   0.0f, -1.0f, 0.0f);
	/* Invierte el eje X para que cuadre con la visión que se ha tomado a lo largo
 	del desarrollo (el origen de coordenadas está en la esquina superior
	izquierda) */
	glScalef (-1, 1, 1);

	glTranslatef (-(x / 2.0f) , -(y / 2.0f), 0.0f);

	for (i = 0; i < malla.dimens.filas; i++)
	{
		for (j = 0; j < malla.dimens.columnas; j++)
		{
			dibujar_casilla ((j * (espacio + lado)),
					 (i * (espacio + lado)),
					 j, i);
		}
	}

	glutSwapBuffers ();

	/* Comprueba los huecos si ha habido algún cambio */
	if (cambio)
	{
		eliminar_coincidencias (&malla);

		imprimir (DETALLE_DEBUG, "Estado de la matriz tras eliminar"
					 " coincidencias: \n");
		if (ver_nv_detalle () >= DETALLE_DEBUG)
		{
			mostrar_malla (malla);
		}

		llenar_vacios (&malla);

		cambio = false;
	}
}

/**
 * Controla el evento al redimensionar la ventana.
 *
 * @param w
 * 		Nuevo ancho de la pantalla.
 *
 * @param h
 * 		Nueva altura de la pantalla.
 */
void manejador_redim (int w, int h)
{

	// Sacado de http://www.lighthouse3d.com/tutorials/glut-tutorial/

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (h == 0)
		h = 1;

	float ratio = 1.0 * w / h;
	// Use the Projection Matrix
	glMatrixMode (GL_PROJECTION);
	// Reset Matrix
	glLoadIdentity ();

	// Set the viewport to be the entire window
	glViewport (0, 0, w, h);

	// Set the correct perspective.
	gluPerspective (45, ratio, 1, 100);

	// Get Back to the Modelview
	glMatrixMode (GL_MODELVIEW);

	imprimir (DETALLE_EXTRA,
		 "%s() --> Ventana redimensionada a %d x %d. Nuevo ratio = %f\n",
		  __FUNCTION__, w, h, ratio);
}

/**
 * Obtiene la columna (más o menos) a la que pertenecen las coordenadas x e y
 * (respectivas al juego, según se usan en OpenGL).
 *
 * @param x
 * 		Posición X del ratón (convertida a coordenada OpenGL).
 *
 * @return
 * 		La columna, o -1 si está fuera de la matriz.
 */
int obtener_col (float x)
{
	/* Dimensiones de la ventana */
	float col = 0,
	      aux = 0.0f;
	bool par = (malla.dimens.columnas % 2) == 0;

	/* Obtiene la celda sobre la que se ha pulsado (si es que
 	 se pulsó sobre alguna) */
	aux = x;
	/* Si el número de columnas es impar, la columna central ocupa desde
 	-PASO_X/2 hasta PASO_X/2 */
	if (!par)
	{
		aux += (x < 0)? -(PASO_X / 2) : (PASO_X / 2);
	}

	col = (x < 0)? floor (aux / PASO_X) : ceil (aux / PASO_X);

	/* Ajusta el índice para coincidir con la notación: de 0 a (n-1) */
	if (col < 0)
	{
		col += (par)? floor (malla.dimens.columnas / 2)
			    : floor (malla.dimens.columnas / 2) + 1;
	}
	else
	{
		col += (floor (malla.dimens.columnas / 2)) - 1;
	}

	if ((col >= malla.dimens.columnas) || (col < 0))
	{
		return -1;
	}

	return col;
}


/**
 * Obtiene la fila (más o menos) a la que pertenecen las coordenadas x e y
 * (respectivas al juego, según se usan en OpenGL).
 *
 * @param y
 * 		Posición Y del ratón (convertida a coordenada OpenGL).
 *
 *
 * @return
 * 		La fila, o -1 si está fuera de la matriz.
 */
int obtener_fila (float y)
{
	/* Dimensiones de la ventana */
	float fila = 0,
	      aux = 0.0f;
	bool par = (malla.dimens.filas % 2) == 0;

	/* Obtiene la celda sobre la que se ha pulsado (si es que
 	 se pulsó sobre alguna) */
	aux = y;
	/* Si el número de filas es impar, la fila central ocupa desde
 	-PASO_Y/2 hasta PASO_Y/2 */
	if (!par)
	{
		aux += (y < 0)? -(PASO_Y / 2) : (PASO_Y / 2);
	}

	fila = (y < 0)? floor (aux / PASO_Y) : ceil (aux / PASO_Y);

	/* Ajusta el índice para coincidir con la notación: de 0 a (n-1) */
	if (fila < 0)
	{
		fila += (par)? floor (malla.dimens.filas / 2)
			    : floor (malla.dimens.filas / 2) + 1;
	}
	else
	{
		fila += (floor (malla.dimens.filas / 2)) - 1;
	}

	if ((fila >= malla.dimens.filas) || (fila < 0))
	{
		return -1;
	}


	return fila;
}

/**
 * Obtiene el movimiento seleccionado segúnlas casillas seleccionadas para intercambiar.
 *
 * @param fila
 * 		Fila de la nueva casilla seleccionada.
 *
 * @param col
 * 		Columna de la nueva casilla seleccionada.
 *
 *
 * @return
 * 		Un elemento de tipo MOV_* (definidos en candy.h), o ERR_MOV si no es un
 * 	movimiento aceptado (por ejemplo, se ha seleccionado otra vez la misma casilla).
 */
int obtener_movimiento (int fila, int col)
{
	if ( (fila == casilla_sel.filas)
	    && (col == casilla_sel.columnas)
	)
	{
		return ERR_MOV;
	}

	/* Fila arriba */
	if ( ((fila - casilla_sel.filas) < 0)
	   && (col == casilla_sel.columnas)
	)
	{
		return MOV_ARRIBA;
	}

	/* Fila abajo */
	if ( ((fila - casilla_sel.filas) > 0)
	   && (col == casilla_sel.columnas)
	)
	{
		return MOV_ABAJO;
	}

	/* Columna a la izquierda */
	if ( ((col - casilla_sel.columnas) < 0)
	   && (fila == casilla_sel.filas)
	)
	{
		return MOV_IZQ;
	}

	/* Columna a la derecha */
	if ( ((col - casilla_sel.columnas) > 0)
	   && (fila == casilla_sel.filas)
	)
	{
		return MOV_DER;
	}

	return ERR_MOV;
}


/**
 * Procesa un evento provocado por el ratón.
 *
 * @param boton
 * 		Botón del ratón pulsado.
 *
 * @param estado
 * 		Estado del botón.
 *
 * @param posX
 * 		Posición X del ratón cuando se pulsó el botón.
 *
 * @param posY
 * 		Posición Y del ratón cuando se pulsó el botón.
 */
void manejador_raton (int boton, int estado, int posX, int posY)
{
	float x = (float) posX,
	      y = (float) posY;

	int columna,
	    fila,
	    mov = ERR_MOV;

	/* Botón liberado */
	if (estado == GLUT_UP)
	{
		return;
	}

	CONVERTIR_COORD_RATON (x, y);

	imprimir (DETALLE_EXTRA,
		  "%s() --> Evento de ratón: %d - Estado: %d - X: %d - Y: %d.\n"
		  "\tPosición respecto a la ventana (GLUT) -> x: %f, y: %f\n",
		  __FUNCTION__, boton, estado, posX, posY,
		  x, y);

	columna = obtener_col (x);
	fila = obtener_fila (y);

	/* Si se ha pinchado dentro de la matriz, selecciona la casilla */
	if ((columna != -1) && (fila != -1))
	{
		imprimir (DETALLE_DEBUG,
			  "Seleccionada la casilla [%d : %d]\n",
			  fila, columna);
		/* Si había alguna casilla seleccionada, las intercambia */
		if ( (casilla_sel.columnas != -1) && (casilla_sel.filas != -1) )
		{
			mov = obtener_movimiento (fila, columna);
			if (mov != ERR_MOV)
			{
				imprimir (DETALLE_DEBUG,
					  "Movimiento de la casilla [%d : %d]\n",
					  casilla_sel.filas, casilla_sel.columnas);

				mover_diamante (casilla_sel.filas, casilla_sel.columnas,
						mov, malla);
				cambio = true;
				/* Reinicia la casilla seleccionada */
				casilla_sel.filas = -1;
				casilla_sel.columnas = -1;
			}
		}
		else
		{
			casilla_sel.filas = fila;
			casilla_sel.columnas = columna;
		}
	}
}

/**
 * Procesa las teclas pulsadas durante el juego.
 *
 * @param tecla
 * 		Código de la tecla pulsada.
 *
 * @param x
 * 		Posición X del ratón respecto a la ventana de juego
 * 	cuando se pulsó la tecla.
 *
 * @param y
 * 		Posición Y del ratón respecto a la ventana de juego
 * 	cuando se pulsó la tecla.
 */
void manejador_teclas (unsigned char tecla, int x, int y)
{
	imprimir (DETALLE_EXTRA,
		  "%s() --> Tecla presionada: %c - X: %d - Y: %d\n",
		  __FUNCTION__, tecla, x, y);

	/* Comprueba si se trata de una bomba */
	if (bomba_act && (sel_bomba > 0)
		&& ( (tecla == '0') || (tecla == '1') || (tecla == '2')
		   || (tecla == '3') || (tecla == '4') || (tecla == '5')
		   || (tecla == '6') || (tecla == '7') || (tecla == '8')
		   || (tecla == '9')
		)
	)
	{
		/* Elimina la fila o columna seleccionada, en función de la bomba que
 		 estuviera activa */
		switch (sel_bomba)
		{
			case 1:
				imprimir (DETALLE_DEBUG,
				  "Bomba para eliminar la fila %c\n", tecla);
				/* Llama a la función que utiliza CUDA */
				bomba_fila (char_to_int (tecla), &malla);
				break;
			case 2:
				imprimir (DETALLE_DEBUG,
					  "Bomba para eliminar la columna %c\n",
					   tecla);
				/* Llama a la función que utiliza CUDA */
				bomba_columna (char_to_int (tecla), &malla);
				break;
		}

		cambio = true;

		bomba_act = false;
		sel_bomba = -1;
		return;
	}

	switch (tecla)
	{
		case 'q':
		case 27: /* ESC */
			exit (SUCCESS);
			break;

		case '1': /* Bomba 1 (si es que estaba activa la selección de bomba */
			if (bomba_act)
			{
				imprimir (DETALLE_EXTRA, "Seleccionada bomba I.\n");
				sel_bomba = 1;
			}
			break;
		case '2': /* Bomba 2 (si es que estaba activa la selección de bomba */
			if (bomba_act)
			{
				imprimir (DETALLE_EXTRA, "Seleccionada bomba II.\n");
				sel_bomba = 2;
			}
			break;
		case '3': /* Bomba 3 (si es que estaba activa la selección de bomba */
			if (bomba_act)
			{
				cambio = true;

				imprimir (DETALLE_EXTRA, "Seleccionada bomba III.\n");
				bomba_giro (&malla);
				bomba_act = false;
			}
			break;
		case '9': /* Activa la selección de bomba */
			bomba_act = true;
			break;

		case 'g':
			/* Guarda en el archivo por defecto */
			if (guardar (malla, ARCHIVO_PARTIDA_OPENGL) == SUCCESS)
			{
				imprimir (DETALLE_LOG,
					  "Partida guardada con éxito en '%s'\n",
					  ARCHIVO_PARTIDA_OPENGL);
			}
			break;

		case 'c':
			/* Carga la partida desde el archivo por defecto */
			if (cargar (&malla, ARCHIVO_PARTIDA_OPENGL) == SUCCESS)
			{
				imprimir (DETALLE_LOG,
					 "Partida cargada con éxito desde '%s'\n",
					 ARCHIVO_PARTIDA_OPENGL);
			}
			break;
	}
}

/* --- Fin de las funciones para OpenGL ----  */

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
