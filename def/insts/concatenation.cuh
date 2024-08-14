#pragma once

#include "insts.cuh"

#define concatenation__Xs 1
#define concatenation__Ys 1
#define concatenation__PARAMS 4
#define concatenation__Nom "concatenation"

uint concatenation__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint concatenation__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void concatenation__init_poids(Inst_t * inst);

void concatenation__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void concatenation__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void concatenation__pre_f(Inst_t * inst);

static fonctions_insts_t fi_concatenation = {
	.Xs    =concatenation__Xs,
	.PARAMS=concatenation__PARAMS,
	.Nom   =concatenation__Nom,
	//
	.calculer_P=concatenation__calculer_P,
	.calculer_L=concatenation__calculer_L,
	//
	.init_poids=concatenation__init_poids,
	//
	.f =concatenation__f,
	.df=concatenation__df,
	//
	.pre_f=concatenation__pre_f
};