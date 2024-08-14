#pragma once

#include "insts.cuh"

#define mul__Xs 2
#define mul__Ys 1
#define mul__PARAMS 0
#define mul__Nom "Mul"

uint mul__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint mul__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void mul__init_poids(Inst_t * inst);

void mul__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void mul__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void mul__pre_f(Inst_t * inst);

static fonctions_insts_t fi_mul = {
	.Xs    =mul__Xs,
	.PARAMS=mul__PARAMS,
	.Nom   =mul__Nom,
	//
	.calculer_P=mul__calculer_P,
	.calculer_L=mul__calculer_L,
	//
	.init_poids=mul__init_poids,
	//
	.f =mul__f,
	.df=mul__df,
	//
	.pre_f=mul__pre_f
};