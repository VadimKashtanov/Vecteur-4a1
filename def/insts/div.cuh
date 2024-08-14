#pragma once

#include "insts.cuh"

#define div__Xs 2
#define div__Ys 1
#define div__PARAMS 0
#define div__Nom "Div"

uint div__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint div__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void div__init_poids(Inst_t * inst);

void div__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void div__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void div__pre_f(Inst_t * inst);

static fonctions_insts_t fi_div = {
	.Xs    =div__Xs,
	.PARAMS=div__PARAMS,
	.Nom   =div__Nom,
	//
	.calculer_P=div__calculer_P,
	.calculer_L=div__calculer_L,
	//
	.init_poids=div__init_poids,
	//
	.f =div__f,
	.df=div__df,
	//
	.pre_f=div__pre_f
};