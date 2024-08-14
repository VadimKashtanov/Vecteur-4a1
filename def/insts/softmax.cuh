#pragma once

#include "insts.cuh"

#define softmax__Xs 1
#define softmax__PARAMS 1 // Vect_Taille
#define softmax__Nom "Softmax"

uint softmax__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint softmax__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void softmax__init_poids(Inst_t * inst);

void softmax__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void softmax__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void softmax__pre_f(Inst_t * inst);

static fonctions_insts_t fi_softmax = {
	.Xs    =softmax__Xs,
	.PARAMS=softmax__PARAMS,
	.Nom   =softmax__Nom,
	//
	.calculer_P=softmax__calculer_P,
	.calculer_L=softmax__calculer_L,
	//
	.init_poids=softmax__init_poids,
	//
	.f =softmax__f,
	.df=softmax__df,
	//
	.pre_f=softmax__pre_f
};