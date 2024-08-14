#pragma once

#include "insts.cuh"

#define union__Xs 2
#define union__PARAMS 0
#define union__Nom "Union"

uint union__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint union__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void union__init_poids(Inst_t * inst);

void union__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void union__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void union__pre_f(Inst_t * inst);

static fonctions_insts_t fi_union = {
	.Xs    =union__Xs,
	.PARAMS=union__PARAMS,
	.Nom   =union__Nom,
	//
	.calculer_P=union__calculer_P,
	.calculer_L=union__calculer_L,
	//
	.init_poids=union__init_poids,
	//
	.f =union__f,
	.df=union__df,
	//
	.pre_f=union__pre_f
};