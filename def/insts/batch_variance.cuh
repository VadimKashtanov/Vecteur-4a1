#pragma once

// var = sum( (e - miu)**2 ) / len(l)

#include "insts.cuh"

#define batch_variance__Xs 2
#define batch_variance__Ys 1
#define batch_variance__PARAMS 1 //C0
#define batch_variance__Nom "Batch batch_variance"

uint batch_variance__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint batch_variance__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void batch_variance__init_poids(Inst_t * inst);

void batch_variance__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void batch_variance__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void batch_variance__pre_f(Inst_t * inst);

static fonctions_insts_t fi_batch_variance = {
	.Xs    =batch_variance__Xs,
	.PARAMS=batch_variance__PARAMS,
	.Nom   =batch_variance__Nom,
	//
	.calculer_P=batch_variance__calculer_P,
	.calculer_L=batch_variance__calculer_L,
	//
	.init_poids=batch_variance__init_poids,
	//
	.f =batch_variance__f,
	.df=batch_variance__df,
	//
	.pre_f=batch_variance__pre_f
};