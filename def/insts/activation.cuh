#pragma once

#include "insts.cuh"

#define activation__Xs 1
#define activation__Ys 1
#define activation__PARAMS 1
#define activation__Nom "Activ"

uint activation__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint activation__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void activation__init_poids(Inst_t * inst);

void activation__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void activation__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void activation__pre_f(Inst_t * inst);

static fonctions_insts_t fi_activation = {
	.Xs    =activation__Xs,
	.PARAMS=activation__PARAMS,
	.Nom   =activation__Nom,
	//
	.calculer_P=activation__calculer_P,
	.calculer_L=activation__calculer_L,
	//
	.init_poids=activation__init_poids,
	//
	.f =activation__f,
	.df=activation__df,
	//
	.pre_f=activation__pre_f
};