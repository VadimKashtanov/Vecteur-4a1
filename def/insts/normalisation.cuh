#pragma once

#include "insts.cuh"

#define normalisation__Xs 2
#define normalisation__Ys 1
#define normalisation__PARAMS 1
#define normalisation__Nom "normalisation"

uint normalisation__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint normalisation__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void normalisation__init_poids(Inst_t * inst);

void normalisation__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void normalisation__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void normalisation__pre_f(Inst_t * inst);

static fonctions_insts_t fi_normalisation = {
	.Xs    =normalisation__Xs,
	.PARAMS=normalisation__PARAMS,
	.Nom   =normalisation__Nom,
	//
	.calculer_P=normalisation__calculer_P,
	.calculer_L=normalisation__calculer_L,
	//
	.init_poids=normalisation__init_poids,
	//
	.f =normalisation__f,
	.df=normalisation__df,
	//
	.pre_f=normalisation__pre_f
};