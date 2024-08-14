#pragma once

#include "insts.cuh"

#define div_scal__Xs 2
#define div_scal__Ys 1
#define div_scal__PARAMS 1
#define div_scal__Nom "Div Scalair"

uint div_scal__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint div_scal__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void div_scal__init_poids(Inst_t * inst);

void div_scal__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void div_scal__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void div_scal__pre_f(Inst_t * inst);

static fonctions_insts_t fi_div_scal = {
	.Xs    =div_scal__Xs,
	.PARAMS=div_scal__PARAMS,
	.Nom   =div_scal__Nom,
	//
	.calculer_P=div_scal__calculer_P,
	.calculer_L=div_scal__calculer_L,
	//
	.init_poids=div_scal__init_poids,
	//
	.f =div_scal__f,
	.df=div_scal__df,
	//
	.pre_f=div_scal__pre_f
};