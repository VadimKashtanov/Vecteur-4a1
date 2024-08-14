#pragma once

#include "insts.cuh"

#define sous__Xs 2
#define sous__Ys 1
#define sous__PARAMS 0
#define sous__Nom "Soustraction"

uint sous__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint sous__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void sous__init_poids(Inst_t * inst);

void sous__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void sous__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void sous__pre_f(Inst_t * inst);

static fonctions_insts_t fi_sous = {
	.Xs    =sous__Xs,
	.PARAMS=sous__PARAMS,
	.Nom   =sous__Nom,
	//
	.calculer_P=sous__calculer_P,
	.calculer_L=sous__calculer_L,
	//
	.init_poids=sous__init_poids,
	//
	.f =sous__f,
	.df=sous__df,
	//
	.pre_f=sous__pre_f
};