#pragma once

#include "insts.cuh"

#define positionnal__Xs 1
#define positionnal__PARAMS 2
#define positionnal__Nom "Positionnal"

uint positionnal__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint positionnal__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void positionnal__init_poids(Inst_t * inst);

void positionnal__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void positionnal__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void positionnal__pre_f(Inst_t * inst);

static fonctions_insts_t fi_positionnal = {
	.Xs    =positionnal__Xs,
	.PARAMS=positionnal__PARAMS,
	.Nom   =positionnal__Nom,
	//
	.calculer_P=positionnal__calculer_P,
	.calculer_L=positionnal__calculer_L,
	//
	.init_poids=positionnal__init_poids,
	//
	.f =positionnal__f,
	.df=positionnal__df,
	//
	.pre_f=positionnal__pre_f
};