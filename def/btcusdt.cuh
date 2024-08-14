#pragma once

#include "meta.cuh"

#define  score_p2(y,w,C) (10*powf(y-w, C  )/(float)C )
#define dscore_p2(y,w,C) (10*powf(y-w, C-1)          )

//	------------------------------------

typedef struct {
	//
	uint I;
	//
	uint T;
	//
	uint L;
	uint N;
	//

	uint X; uint Y;

	//	Espaces
	float * serie__d;	//	I * T * L * N
} BTCUSDT_t;

BTCUSDT_t * cree_btcusdt(char * fichier);
void     liberer_btcusdt(BTCUSDT_t * btcusdt);
//
float* pourcent_btcusdt(BTCUSDT_t * btcusdt, float * y__d, uint * ts__d);
//
float f_df_btcusdt(BTCUSDT_t * btcusdt, float * y__d, float * dy__d, uint * ts__d);