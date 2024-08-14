#pragma once

#include "insts.cuh"

#define transpose2d__Xs 1
#define transpose2d__Ys 1
#define transpose2d__PARAMS 3
#define transpose2d__Nom "Transpose 2d"

uint transpose2d__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint transpose2d__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void transpose2d__init_poids(Inst_t * inst);

void transpose2d__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void transpose2d__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void transpose2d__pre_f(Inst_t * inst);

static fonctions_insts_t fi_transpose2d = {
	.Xs    =transpose2d__Xs,
	.PARAMS=transpose2d__PARAMS,
	.Nom   =transpose2d__Nom,
	//
	.calculer_P=transpose2d__calculer_P,
	.calculer_L=transpose2d__calculer_L,
	//
	.init_poids=transpose2d__init_poids,
	//
	.f =transpose2d__f,
	.df=transpose2d__df,
	//
	.pre_f=transpose2d__pre_f
};