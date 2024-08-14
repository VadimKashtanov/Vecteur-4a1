#pragma once

#include "insts.cuh"

#define matmul__Xs 2
#define matmul__Ys 1
#define matmul__PARAMS 4
#define matmul__Nom "Mat mul"

uint matmul__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint matmul__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void matmul__init_poids(Inst_t * inst);

void matmul__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void matmul__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void matmul__pre_f(Inst_t * inst);

static fonctions_insts_t fi_matmul = {
	.Xs    =matmul__Xs,
	.PARAMS=matmul__PARAMS,
	.Nom   =matmul__Nom,
	//
	.calculer_P=matmul__calculer_P,
	.calculer_L=matmul__calculer_L,
	//
	.init_poids=matmul__init_poids,
	//
	.f =matmul__f,
	.df=matmul__df,
	//
	.pre_f=matmul__pre_f
};