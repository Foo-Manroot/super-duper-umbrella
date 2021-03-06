#ifndef _CANDY_H_
#define _CANDY_H_

#include "common.h"
#include "libutils.h"
#include "kernel.cuh"

#include <sys/time.h>

#ifdef __APPLE__
	#include <GLUT/freeglut.h>
#else
	#include <GL/freeglut.h>
#endif


/**
 * Macro para convertir la coordenada X en la pantalla en coordenada respecto a la
 * ventana (para GLUT).
 */
#define CONVERTIR_COORD_RATON(x, y)						\
	x = ( ((float) x) / ((float) glutGet (GLUT_WINDOW_WIDTH))  ) - 0.5f;	\
	y = ( ((float) y) / ((float) glutGet (GLUT_WINDOW_HEIGHT)) ) - 0.5f;


/**
 * Macro para convertir la coordenada X con respecto a la ventana (GLUT) en coordenada
 * respecto a la pantalla.
 */
#define CONVERTIR_COORD_GLUT(x, y)						\
	x = ( ((float) x) + 0.5f * ((float) glutGet (GLUT_WINDOW_WIDTH))  );	\
	y = ( ((float) y) + 0.5f * ((float) glutGet (GLUT_WINDOW_HEIGHT)) );

/**
 * Archivo por defecto para guardar/cargar partidas
 */
#define ARCHIVO_PARTIDA_OPENGL "partida.asdf.pepe"


/**
 * Umbral para controlar los FPS de manera chapucera
 */
#define UMBRAL_FPS 100000

/**
 * En las coordenadas de GLUT, el lado de los cuadrados parece ser este (sacado a ojo)
 */
#define PASO_X 0.0673
#define PASO_Y 0.1346


/**
 * Movimientos de diamantes posibles
 */
#define MOV_DER 0
#define MOV_ABAJO 1
#define MOV_IZQ 2
#define MOV_ARRIBA 3

/* ------------------------ */
/* DECLARACIÓN DE FUNCIONES */
/* ------------------------ */
void mover_diamante(int posY, int posX, int mov, Malla malla);

int es_valido(int posY, int posX, int mov, Malla malla);

/**
 * Se encarga de todo lo necesario para iniciar la ventana con OpenGL.
 */
void iniciar_opengl (int argc, char *argv []);

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
void dibujar_casilla (float x, float y, int posX, int posY);

/**
 * Dibuja los elementos en la pantalla.
 */
void render (void);

/**
 * Controla el evento al redimensionar la ventana.
 *
 * @param w
 * 		Nuevo ancho de la pantalla.
 *
 * @param h
 * 		Nueva altura de la pantalla.
 */
void manejador_redim (int w, int h);

/**
 * Dibuja los elementos en la pantalla.
 */
void manejador_gui (void);

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
void manejador_teclas (unsigned char tecla, int x, int y);

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
int obtener_col (float x);

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
int obtener_fila (float y);
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
int obtener_movimiento (int fila, int col);

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
void manejador_raton (int boton, int estado, int posX, int posY);

#endif
